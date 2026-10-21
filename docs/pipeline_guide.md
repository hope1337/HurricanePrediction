# Pipeline dự báo quỹ đạo bão

## 1. Dự án muốn làm gì?

**Dùng lịch sử di chuyển của một cơn bão để dự báo vị trí tâm bão trong 48 giờ tiếp theo.** Phạm vi là bão ở Tây Thái Bình Dương, gồm khu vực Philippines, Nhật Bản và Biển Đông.

Ví dụ: tại 00:00 hôm nay, ta đã biết bão ở đâu trong những giờ vừa qua. Mô hình nhận lịch sử đó và đoán vị trí lúc 06:00, 12:00, …, đến 00:00 hai ngày sau — tổng cộng **8 vị trí, cách nhau 6 giờ**.

Dự án dùng dữ liệu bão trong quá khứ để học và kiểm tra khả năng dự báo. Đầu vào hiện tại chỉ có vị trí và thông tin chuyển động tính từ vị trí; đầu ra là quỹ đạo tâm bão, không phải sức gió hay lượng mưa.

Dự án so sánh ba loại mô hình học từ chuỗi dữ liệu theo thời gian: **RNN, GRU và LSTM**. Với mỗi loại, ta thử hai cách dự báo: đoán cả 48 giờ trong một lần, hoặc đoán từng bước 6 giờ rồi dùng kết quả đó để đoán tiếp. Ngoài ra, dự án kiểm tra xem việc cung cấp thêm tốc độ và hướng di chuyển có giúp mô hình dự báo tốt hơn so với chỉ cho biết vị trí hay không.

Vì vậy, kết quả cần thu được không chỉ là một đường dự báo trên bản đồ. Ta còn cần những phép so sánh để biết cách làm nào hiệu quả hơn, sai số là bao nhiêu và kết luận đó được rút ra từ dữ liệu nào.

## 2. Toàn bộ luồng xử lý

**Pipeline** là chuỗi bước biến dữ liệu gốc thành dự báo và số liệu đánh giá:

```text
Lịch sử vị trí bão (.nc)
    → Lọc và tách các đoạn liên tục, cách nhau 6 giờ
    → Chia train / validation / test theo năm của bão
    → Tạo mẫu: lịch sử đầu vào + 8 vị trí tương lai làm đáp án
    → Chuẩn hóa đầu vào, lưu dữ liệu đã xử lý (.npz)
    → Huấn luyện mô hình, giữ bản tốt nhất trên validation (.pt)
    → Dự báo trên test, so với vị trí thật → sai số tính bằng km
```

Toàn bộ dữ liệu ở đây đều là lịch sử đã được ghi nhận, nên ta biết cả vị trí trước và sau một thời điểm dự báo. Để mô phỏng tình huống dự báo thật, code chỉ đưa phần lịch sử vào mô hình. Các vị trí phía sau được tách riêng thành **đáp án**.

Khi huấn luyện, mô hình dự báo rồi so với đáp án để điều chỉnh cách học. Khi kiểm tra, mô hình vẫn chỉ được thấy lịch sử; đáp án chỉ dùng sau khi dự báo xong để chấm điểm. Đây là lý do cùng một bộ dữ liệu có thể dùng để xây dựng bài toán dự báo tương lai mà không cần đợi một cơn bão mới xuất hiện.

## 3. Dữ liệu được chuẩn bị thế nào?

### Giữ các quan trắc phù hợp

Dữ liệu gốc nằm ở `data/IBTrACS.WP.v04r01.nc`, lấy từ bộ dữ liệu lịch sử bão **NOAA IBTrACS**. Một quan trắc là một lần ghi nhận bão: mã bão (`SID`), thời gian, vĩ độ (`LAT`) và kinh độ (`LON`). Các dòng cùng mã bão tạo thành lịch sử của cơn bão đó.

Pipeline giữ dữ liệu năm **1980–2024**, thuộc vùng Tây Thái Bình Dương (`WP`) và có loại quỹ đạo chính (`main`). Các bản ghi thuộc nhánh quỹ đạo phụ hoặc dữ liệu tạm thời bị loại theo phạm vi xử lý của dự án.

Tiếp theo, code chỉ giữ các mốc **00:00, 06:00, 12:00, 18:00 UTC**, rồi bỏ vị trí không hợp lệ, sắp xếp theo thời gian và loại dòng trùng. Việc chọn các mốc này giúp mỗi bước của chuỗi tương ứng với 6 giờ. Nếu trộn cả quan trắc cách nhau 3 giờ và 6 giờ, cùng một bước dữ liệu sẽ mang hai ý nghĩa thời gian khác nhau, khiến bài toán khó nhất quán.

Việc lọc vùng biển áp dụng cho **từng quan trắc**: nếu bão đi ra ngoài WP, phần ở ngoài bị bỏ. Nếu hai quan trắc còn lại không cách nhau đúng 6 giờ, chuỗi bị tách thành các **đoạn liên tục** (*segment*). Chẳng hạn, nếu có vị trí lúc 06:00 và 18:00 nhưng thiếu 12:00, hai vị trí này không được coi là hai bước liền nhau. Code cắt đoạn tại chỗ đó thay vì giả định bão đã đi thế nào trong khoảng bị thiếu.

### Chia dữ liệu để học và kiểm tra

| Tập | Năm của bão | Vai trò | Số mẫu hiện có |
|---|---|---|---:|
| Train | 1980–2016 | Cho mô hình học từ ví dụ | 27.719 |
| Validation | 2017–2020 | Chọn phiên bản mô hình tốt nhất trong lúc học | 2.499 |
| Test | 2021–2024 | Đánh giá mô hình sau khi đã chọn xong | 2.207 |

Có thể hiểu train là phần bài tập để học, validation là phần kiểm tra trong quá trình học, còn test là bài đánh giá cuối cùng. Validation giúp chọn phiên bản mô hình tốt nhất; test cho biết phiên bản đã chọn làm tốt đến đâu trên các cơn bão ở giai đoạn sau.

**Một cơn bão chỉ thuộc một tập**, dựa trên năm của cơn bão đó. Ta không chia ngẫu nhiên các dòng rồi để nửa đầu một cơn bão vào train, nửa sau vào test. Các vị trí gần nhau trong cùng một cơn bão có liên hệ rất mạnh; chia như vậy có thể khiến kết quả trông tốt hơn khả năng dự báo trên bão chưa từng thấy. Các con số trong bảng là số cửa sổ dự báo được tạo ra, không phải số cơn bão.

### Cắt mỗi đoạn thành các mẫu dự báo

Một **mẫu** (*window*, hay cửa sổ) gồm **14 vị trí liên tiếp**: 6 vị trí lịch sử và 8 vị trí tương lai làm đáp án. Gọi thời điểm bắt đầu dự báo là `t`:

| Phần | Các mốc thời gian | Dùng để làm gì? |
|---|---|---|
| 1 vị trí phụ | `t−30h` | Tính chuyển động cho hàng đầu vào đầu tiên |
| 5 vị trí đưa vào mô hình | `t−24h, t−18h, t−12h, t−6h, t` | Lịch sử để dự báo |
| 8 vị trí tương lai | `t+6h, t+12h, …, t+48h` | Đáp án để học hoặc chấm điểm |

**Vì sao cần 6 vị trí lịch sử nhưng chỉ đưa vào 5 hàng?** Muốn biết bão di chuyển thế nào ở mốc `t−24h`, cần so vị trí lúc đó với vị trí trước nó (`t−30h`). Vị trí phụ chỉ phục vụ phép tính này.

Cửa sổ trượt mỗi lần một mốc 6 giờ để tạo mẫu tiếp theo. Ví dụ, đánh số 16 vị trí trong một đoạn từ 0 đến 15: mẫu đầu dùng vị trí 0–13, mẫu thứ hai dùng 1–14 và mẫu thứ ba dùng 2–15. Trong mỗi mẫu, 6 vị trí đầu là lịch sử và 8 vị trí sau là đáp án. Các mẫu có thể chồng lấn vì đó là những lần bắt đầu dự báo ở các thời điểm khác nhau.

Với đoạn dài `L` vị trí, số mẫu là `max(0, L − 13)`. Đoạn ngắn hơn 14 vị trí không tạo được mẫu; một mẫu luôn nằm trong cùng một đoạn của cùng một cơn bão. Cách cắt mẫu này dùng chung cho cả hai chiến lược dự báo để chúng được kiểm tra trên cùng những tình huống.

### Mỗi hàng đầu vào chứa gì?

Một **đặc trưng** (*feature*) là một con số mô tả bão tại một mốc thời gian. Dự án thử ba cách chọn đặc trưng:

| Cấu hình | Thông tin tại mỗi mốc | Số đặc trưng |
|---|---|---:|
| A | Vĩ độ, kinh độ | 2 |
| B | Vĩ độ, kinh độ, tốc độ, hướng biểu diễn bằng sin/cos | 5 |
| **C — cấu hình chính** | Vĩ độ, kinh độ, vận tốc theo hướng đông và hướng bắc | 4 |

Tốc độ và hướng được **tự tính từ hai vị trí liên tiếp đã biết**, không lấy cột chuyển động có sẵn trong file. Như vậy, đặc trưng tại thời điểm nào cũng chỉ phụ thuộc vào vị trí ở thời điểm đó và trước đó, tránh đưa thông tin tương lai vào đầu vào.

Gọi `d` là khoảng cách bão đã đi trong 6 giờ, tính bằng km, và `θ` là hướng di chuyển. Hướng được đo từ phía bắc theo chiều kim đồng hồ: bắc là 0°, đông là 90°, nam là 180° và tây là 270°. Ta tính:

```text
v   = d / 6                 (tốc độ, km/h)
v_E = v × sin(θ)            (thành phần vận tốc theo hướng đông)
v_N = v × cos(θ)            (thành phần vận tốc theo hướng bắc)
```

Ví dụ minh họa, nếu bão đi được 120 km trong 6 giờ theo hướng 60°, tốc độ là `120 / 6 = 20 km/h`. Hai thành phần vận tốc là `20 × sin(60°) ≈ 17,3 km/h` về đông và `20 × cos(60°) = 10 km/h` về bắc. Nếu vị trí hiện tại là 15,5°B, 131°Đ, hàng đầu vào C trước chuẩn hóa sẽ là `[15.5, 131.0, 17.3, 10.0]`. Thành phần âm tương ứng với hướng tây hoặc nam.

Ở cấu hình B, hướng được biểu diễn bằng cặp `sin(θ), cos(θ)`. Lý do là 359° và 1° chỉ lệch nhau 2° trên la bàn, nhưng nếu đưa hai số góc thô vào mô hình thì chúng lại chênh nhau tới 358. Cặp sin/cos giữ được tính chất vòng tròn của hướng.

Khoảng cách và hướng giữa hai tọa độ được tính bằng `pyproj.Geod` trên mô hình Trái Đất WGS84; code có phương án thay thế trên mặt cầu khi thư viện này không có. Không lấy hiệu kinh độ, vĩ độ rồi coi đó là km, vì một độ kinh độ ứng với khoảng cách khác nhau tùy vĩ độ. Kinh độ đưa vào mô hình cũng được làm liên tục trong từng đoạn để tránh bước nhảy số khi đi qua đường đổi ngày; tọa độ đáp án vẫn giữ quy ước 0–360°Đ của dữ liệu gốc.

### Chuẩn hóa trước khi đưa vào mô hình

Vĩ độ, kinh độ và vận tốc có thang giá trị khác nhau. Để mô hình dễ học hơn, mỗi cột đầu vào được **chuẩn hóa** bằng công thức:

```text
x_chuẩn_hóa = (x − μ_train) / σ_train
```

Trong đó, `μ_train` là trung bình và `σ_train` là độ lệch chuẩn của cột đó trên các hàng đầu vào của tập train. Ví dụ minh họa, nếu vĩ độ có trung bình 20° và độ lệch chuẩn 10°, giá trị 15° sẽ thành `(15 − 20) / 10 = −0,5`. Đây là cách đổi thang số, không phải thay đổi vị trí bão.

Validation và test dùng nguyên bộ trung bình, độ lệch chuẩn đã tính từ train. Nếu tính lại bằng dữ liệu test, bước chuẩn bị đã sử dụng thông tin của phần đáng lẽ chỉ được dùng để đánh giá. Bộ giá trị chuẩn hóa được lưu trong `scalers.joblib`; các đáp án tọa độ và độ dịch chuyển không được chuẩn hóa.

Với cấu hình C, ký hiệu đầu vào `[N, 5, 4]` nghĩa là **N mẫu × 5 mốc thời gian × 4 đặc trưng**. Đáp án tọa độ có dạng `[N, 8, 2]`: 8 vị trí, mỗi vị trí gồm vĩ độ và kinh độ.

## 4. Mô hình học và dự báo ra sao?

Dự án ghép **3 loại mô hình (RNN, GRU, LSTM)** với **2 cách dự báo**, tạo thành 6 cấu hình so sánh. Cả ba loại đều đọc lịch sử theo thứ tự thời gian để học quy luật di chuyển.

| Cách dự báo | Cách hoạt động | Điểm cần hiểu |
|---|---|---|
| **Direct** | Nhận lịch sử → đoán cả 8 vị trí trong một lần | Mỗi vị trí tương lai được xác định so với vị trí cuối đã biết |
| **Recursive** | Đoán vị trí sau 6h → đưa dự báo đó vào lịch sử → đoán tiếp, lặp 8 lần | Sai số của bước trước có thể ảnh hưởng bước sau |

### Từ đầu ra mô hình đến tọa độ dự báo

Mô hình thực tế xuất **độ dịch chuyển theo hướng đông/bắc, đơn vị km**. Với direct, cặp `[40, 30]` ở bước +6h có nghĩa là vị trí dự báo nằm cách điểm cuối đã biết 40 km về đông và 30 km về bắc. Nếu bước +12h là `[80, 50]`, cặp này vẫn tính từ điểm cuối đã biết, không phải đi thêm từ vị trí +6h.

Với recursive, mỗi cặp độ dịch chuyển lại tính từ vị trí ngay trước đó. Sau khi đoán +6h, code chuyển kết quả thành tọa độ, tính lại chuyển động, bỏ hàng lịch sử cũ nhất rồi thêm hàng dự báo mới vào. Mô hình dùng cửa sổ đã cập nhật để đoán +12h và tiếp tục đến +48h. Quá trình tự dùng lại dự báo của mình này được gọi là *autoregressive* hoặc *rollout*.

Để đổi cặp độ dịch chuyển `(e, n)` thành tọa độ, code tính quãng đường và hướng:

```text
d = √(e² + n²)
θ = atan2(e, n)
```

Ở đây `atan2(e, n)` cho hướng tính từ phía bắc; góc được đổi sang độ trước khi đưa vào hàm trắc địa. Ví dụ `[40, 30]` tương ứng với quãng đường 50 km theo hướng khoảng 53,1°. Biết điểm xuất phát, quãng đường và hướng, `Geod` tính ra tọa độ đích trên bề mặt Trái Đất. Sau bước chuyển đổi này, cả direct và recursive đều có 8 cặp vĩ độ/kinh độ để so với đáp án thật.

### Mô hình học bằng cách nào?

Trong mỗi lượt học, mô hình nhận một nhóm mẫu, đưa ra dự báo rồi so với đáp án. Hàm **loss** gán một giá trị phạt cho sai số; thuật toán huấn luyện điều chỉnh các tham số bên trong mô hình để giảm giá trị đó. Một lần đi qua toàn bộ tập train được gọi là một **epoch**.

Direct học đồng thời 8 độ dịch chuyển tính từ điểm gốc. Recursive học một bước 6 giờ từ lịch sử thật: mỗi cửa sổ 14 vị trí tạo được 8 bài học một bước, lần lượt dịch lịch sử và đáp án lên một mốc. Vì thế, 27.719 cửa sổ train tạo ra 221.752 mẫu học một bước cho recursive. Những mẫu này có chồng lấn, không phải 221.752 tình huống độc lập.

Dự án dùng **Huber loss**. Với `r` là giá trị dự báo trừ đáp án của một thành phần đông hoặc bắc, công thức là:

```text
             { 0,5 × r²                     nếu |r| ≤ δ
L_δ(r) =     {
             { δ × (|r| − 0,5 × δ)          nếu |r| > δ

δ = 50 km theo cấu hình mặc định
```

Loss của nhóm mẫu là trung bình các giá trị phạt trên những thành phần đầu ra được học. Khi sai số nhỏ, phần bình phương thúc đẩy mô hình sửa sai; khi sai số lớn, phần tuyến tính hạn chế ảnh hưởng quá mạnh của một vài dự báo lệch nhiều. **Huber loss được tính trên các thành phần dịch chuyển; nó không phải khoảng cách địa lý hay ADE** dùng để báo cáo kết quả.

Sau mỗi epoch, code dự báo trên validation rồi tính ADE như giải thích ở phần sau. Recursive lúc này phải tự dự báo đủ 8 bước, không được dùng vị trí thật của các bước tương lai để đi tiếp. Bản có ADE thấp nhất trên validation được lưu thành **`best.pt`**; file lưu trạng thái mô hình như vậy được gọi là *checkpoint*. Mặc định, quá trình học dừng sau tối đa 100 epoch, hoặc sớm hơn nếu 15 epoch liên tiếp không cải thiện.

Hai cách dự báo dùng cùng các trường hợp validation/test để so sánh. Tuy nhiên, recursive có nhiều mẫu học một bước hơn, nên số mẫu train và thời gian học cũng cần được xét khi đọc kết quả.

## 5. Đọc kết quả thế nào?

### Bước đầu: đo sai số của từng vị trí

Trước khi tính các chỉ số tổng hợp (*metric*), ta cần biết mỗi vị trí dự báo lệch vị trí thật bao nhiêu km. Đây là khoảng cách giữa hai điểm trên bề mặt Trái Đất, không phải độ dài đường đi của cơn bão.

Code dùng công thức **Haversine**, coi Trái Đất là mặt cầu có bán kính `R = 6371,0088 km`. Gọi `(φ₁, λ₁)` là vĩ độ, kinh độ dự báo và `(φ₂, λ₂)` là vĩ độ, kinh độ thật. Đổi các góc từ độ sang radian bằng `radian = độ × π / 180`, sau đó tính:

```text
Δφ = φ₂ − φ₁
Δλ = λ₂ − λ₁, được quy về khoảng [−π, π)

a = sin²(Δφ / 2) + cos(φ₁) × cos(φ₂) × sin²(Δλ / 2)
d = 2 × R × arcsin(√a)
```

Kết quả `d` là sai số tính bằng km. Trong code, `a` được chặn vào khoảng `[0, 1]` để tránh lỗi làm tròn số. Ví dụ, nếu dự báo là `(15°B, 131°Đ)` còn vị trí thật là `(16°B, 131°Đ)`, sai số xấp xỉ **111,2 km**. Việc quy chênh lệch kinh độ về khoảng trên giúp xử lý đúng hai điểm nằm gần nhau ở hai phía đường đổi ngày.

Phần tạo đặc trưng và chuyển độ dịch chuyển về tọa độ dùng `Geod` trên WGS84 khi có thư viện; riêng phần chấm metric luôn dùng công thức Haversine này để mọi mô hình được chấm theo cùng một quy tắc.

### MTE, ADE và FDE tính từ các khoảng cách đó

Gọi `N` là số mẫu dự báo và `H = 8` là số bước tương lai. Ký hiệu `d(i, k)` là sai số Haversine của mẫu thứ `i` tại bước thứ `k`, với `k = 1` tương ứng +6h và `k = 8` tương ứng +48h. Sau khi chấm từng vị trí, ta có một bảng **N hàng × 8 cột sai số**.

**MTE (Mean Track Error)** là sai số trung bình tại một mốc tương lai. Ta chọn một cột trong bảng rồi lấy trung bình các mẫu:

```text
MTE(k) = [d(1, k) + d(2, k) + … + d(N, k)] / N
```

Chẳng hạn, MTE(+24h) dùng cột thứ 4 vì `4 × 6 = 24 giờ`. Chỉ số này trả lời: “Nếu dự báo trước 24 giờ, vị trí bão lệch trung bình bao nhiêu km?” Báo cáo thường trình bày các mốc +6h, +12h, +24h và +48h; code vẫn tính đủ cả 8 mốc.

**ADE (Average Displacement Error)** là sai số trung bình trên toàn bộ quỹ đạo dự báo. Ta cộng tất cả ô của bảng rồi chia cho tổng số ô:

```text
ADE = [Σ từ i=1 đến N  Σ từ k=1 đến H  d(i, k)] / (N × H)
```

Ký hiệu `Σ` nghĩa là cộng các giá trị theo chỉ số được ghi bên cạnh. Với ADE, ta cộng sai số ở cả 8 bước của tất cả mẫu. Vì vậy, ADE cho biết chất lượng chung của cả quỹ đạo, không chỉ vị trí cuối. **Không lấy trung bình riêng 4 mốc được trình bày trong báo cáo để thay cho ADE.**

**FDE (Final Displacement Error)** chỉ quan tâm vị trí cuối ở +48h. Ta lấy trung bình cột cuối:

```text
FDE = [d(1, H) + d(2, H) + … + d(N, H)] / N
    = MTE(+48h)
```

FDE = 480 km có nghĩa là sau 48 giờ, vị trí dự báo cách vị trí thật trung bình 480 km trên tập đang đánh giá. Nó không có nghĩa bão đã đi 480 km, hay mọi mẫu đều sai đúng 480 km. Cả MTE, ADE và FDE đều tính bằng **km và càng thấp càng tốt**.

### Ví dụ tính đủ 8 bước

Giả sử đã tính Haversine cho hai mẫu và thu được bảng dưới đây. Đây là số minh họa để tính tay, không phải kết quả thí nghiệm:

| Mẫu | +6h | +12h | +18h | +24h | +30h | +36h | +42h | +48h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Mẫu 1 | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 |
| Mẫu 2 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 |

MTE tại +6h là `(10 + 20) / 2 = 15 km`. Tương tự, MTE tại +24h là `(40 + 50) / 2 = 45 km`. Tổng sai số của mẫu 1 là 360 km, của mẫu 2 là 440 km, nên **ADE = (360 + 440) / (2 × 8) = 50 km**. Riêng cột cuối cho **FDE = (80 + 90) / 2 = 85 km**, cũng chính là MTE tại +48h.

### Trung bình theo mẫu và theo cơn bão khác nhau thế nào?

Các chỉ số chính lấy trung bình theo **cửa sổ dự báo**. Một cơn bão có lịch sử dài tạo được nhiều cửa sổ, nên đóng góp nhiều hơn vào kết quả chung. Bởi vậy, code còn trả về ADE trung bình của các cửa sổ thuộc từng cơn bão để có thể xem riêng.

Ví dụ, bão A có hai cửa sổ với ADE là 40 và 60 km; bão B có một cửa sổ với ADE là 100 km. Trung bình theo cửa sổ là `(40 + 60 + 100) / 3 ≈ 66,7 km`. Nếu muốn mỗi cơn bão có trọng số ngang nhau, trước tiên tính ADE của A là 50 km, của B là 100 km, rồi lấy `(50 + 100) / 2 = 75 km`. Hai số trả lời hai cách tổng hợp khác nhau; báo cáo chính của dự án dùng cách thứ nhất.

Code cũng tính **median** (trung vị) và **P90** ở từng mốc. Median là mức sai số ở giữa khi sắp xếp các mẫu, giúp nhìn kết quả mà ít bị vài mẫu sai rất lớn kéo lên. P90 là phân vị 90% của sai số, mô tả mức lệch ở phía các dự báo kém hơn. Những chỉ số này bổ sung cho trung bình; chúng không thay thế ADE hay FDE.

Khi so các mô hình, phải dùng cùng mã mẫu (`sample_id`), cùng thứ tự và cùng đáp án. Code có kiểm tra điều này khi so sánh tập mẫu. Nếu chủ động loại ô bằng mặt nạ hợp lệ (*valid mask*), mẫu số trong các công thức phải đổi thành số ô thực sự được tính, và số lượng đó phải được báo cáo. Trong trường hợp đầy đủ dữ liệu như ví dụ trên, mẫu số lần lượt là `N`, `N × H` và `N`.

### Kết quả hiện có nói gì?

Trên test, 6 cấu hình chính đạt **ADE khoảng 245–250 km** và **FDE khoảng 479–505 km**. Hiểu theo các định nghĩa trên, vị trí dự báo lệch trung bình khoảng 245–250 km khi xét cả 8 mốc; riêng ở mốc +48h, độ lệch trung bình khoảng 479–505 km.

Trong thí nghiệm thay đặc trưng của LSTM-direct, cấu hình A đạt ADE khoảng 244,1 km, B khoảng 253,0 km và C khoảng 250,2 km. Như vậy, ở lần chạy này, thêm thông tin chuyển động chưa giúp giảm sai số. Đây là kết quả với **seed 0**, tức giá trị dùng để cố định các thao tác ngẫu nhiên khi chạy thí nghiệm; một lần chạy chưa đủ để khẳng định cấu hình nào luôn tốt nhất. Bảng đầy đủ và hình so sánh nằm trong [báo cáo thí nghiệm](../experiments/report.md).

## 6. Chạy pipeline

Chạy lệnh từ **thư mục gốc repo**. Xem [README — cài đặt môi trường](../README.md#1-cài-đặt-môi-trường) nếu chưa cài Python và các thư viện cần thiết.

### Chạy một mô hình từ đầu đến cuối

```bash
conda activate machine-learning

# 1. Dữ liệu gốc → dữ liệu đã xử lý
python scripts/preprocess.py --config configs/data.yaml

# 2. Huấn luyện LSTM với cách dự báo direct, mặc định dùng đặc trưng C
python scripts/train.py --model lstm --strategy direct

# 3. Chấm mô hình tốt nhất trên test, in chỉ số ra terminal
python scripts/evaluate.py --checkpoint experiments/checkpoints/lstm_direct_C_s0/best.pt --split test
```

Lệnh thứ nhất tạo các file đầu vào và đáp án trong `data/processed/`. Lệnh thứ hai học từ train, chọn mô hình bằng validation rồi lưu checkpoint. Lệnh thứ ba nạp `best.pt`, dự báo trên test và in các chỉ số vừa giải thích ở phần 5.

Chỉ cần chạy lại preprocess khi thay dữ liệu hoặc cấu hình xử lý. Nếu đã có dữ liệu đã xử lý và checkpoint tương ứng, có thể chạy thẳng bước đánh giá. Thêm `--out experiments/results/lstm_direct_test.json` vào lệnh đánh giá nếu muốn lưu chỉ số thành file.

### Chạy các thí nghiệm so sánh

Sau khi đã preprocess, chọn các lệnh phù hợp:

```bash
# Huấn luyện và đánh giá đủ 6 cấu hình RNN/GRU/LSTM × direct/recursive
python scripts/run_experiments.py --experiment main

# So sánh đặc trưng A/B/C trên cùng mô hình LSTM-direct (ablation)
python scripts/run_experiments.py --experiment ablation

# Phân tích sai số theo nhóm chuyển động: đi thẳng, đổi hướng, khó phân loại
# Cần có kết quả của main trước
python scripts/run_experiments.py --experiment movement
```

Các lệnh `run_experiments.py` cập nhật bảng CSV, hình và báo cáo từ kết quả đang có. `ablation` là thí nghiệm thay riêng nhóm đặc trưng để xem thông tin nào giúp dự báo tốt hơn.

## 7. Tìm dữ liệu, kết quả và code ở đâu?

| Đường dẫn | Nội dung |
|---|---|
| `configs/data.yaml` | Quy tắc lọc dữ liệu, năm chia tập |
| `configs/training.yaml` | Thiết lập huấn luyện và thí nghiệm |
| `data/processed/` | `train.npz`, `validation.npz`, `test.npz` chứa các mảng đầu vào/đáp án; `scalers.joblib` lưu bộ chuẩn hóa |
| `experiments/checkpoints/` | Mô hình đã học, cấu hình và lịch sử huấn luyện của từng lần chạy |
| `experiments/results/`, `experiments/figures/` | Bảng chỉ số và hình do các thí nghiệm sinh ra |
| `experiments/report.md` | Báo cáo tổng hợp kết quả |
| `src/data/`, `src/models/`, `src/training/`, `src/evaluation/` | Code tương ứng với chuẩn bị dữ liệu, mô hình, huấn luyện và đánh giá |

Đọc sâu hơn khi cần: [tiền xử lý](preprocessing_report.md) · [mô hình](models.md) · [cách tính chỉ số](evaluation.md).
