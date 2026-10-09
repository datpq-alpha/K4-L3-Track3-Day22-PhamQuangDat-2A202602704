# Chấm chéo bằng GPT-4o mini trên máy local

Luồng này dùng lại `data/eval/side_by_side.jsonl` và `judge_results_rm.json` đã lưu từ NB4.
Không cần Colab, GPU, trọng số mô hình hoặc sinh lại câu trả lời. Script dùng cùng prompt,
cách đổi thứ tự A/B, xử lý lỗi và thống kê của `lab22/judge.py` như NB4.

## 1. Cấu hình

Trong thư mục repo, tạo hoặc sửa `.env`:

```dotenv
JUDGE_PROVIDER=openai
JUDGE_MODEL=gpt-4o-mini
OPENAI_API_KEY=
```

Điền khoá OpenAI API của bạn sau dấu `=` ở dòng cuối. `.env` đã được Git bỏ qua;
không đưa khoá vào notebook, ảnh, log hoặc commit. Nếu có biến môi trường cùng tên,
giá trị môi trường được ưu tiên hơn `.env`.

`gpt-4o-mini` là model được chọn cho lần chấm này. Tài liệu chính thức xác nhận model
hỗ trợ Chat Completions và structured outputs:
[GPT-4o mini](https://developers.openai.com/api/docs/models/gpt-4o-mini).
Caller hiện tại dùng JSON mode và `max_completion_tokens=200`, theo
[Chat Completions API](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).
Temperature giữ mặc định của provider, giống caller NB4, nên không khẳng định các lần chấm
sẽ cho kết quả giống tuyệt đối.

Nếu máy chưa có SDK, chỉ cần cài phụ thuộc API:

```powershell
python -m pip install -r requirements-api.txt
```

## 2. Kiểm tra và chạy

Từ thư mục repo, chạy kiểm tra dữ liệu/cấu hình trước; lệnh này không gọi API:

```powershell
python -X utf8 scripts/judge_saved_outputs.py --check
```

Cần thấy model đã chọn, 58 cặp / 50 held-out, `RM match: OK` và
`OPENAI_API_KEY: present`. Bước này chỉ kiểm tra có khoá, chưa xác nhận quyền truy cập
model hoặc hạn mức tài khoản; các thông tin đó được kiểm tra khi gọi API thật.

Chạy thử hai cặp để xác nhận API hoạt động:

```powershell
python -X utf8 scripts/judge_saved_outputs.py --limit 2
```

Sau đó chấm nốt toàn bộ:

```powershell
python -X utf8 scripts/judge_saved_outputs.py
```

Mỗi cặp được chấm hai lần với A/B đảo chỗ, kể cả cặp có hai câu trả lời giống nhau.
58 cặp cần 116 lượt gọi hợp lệ trong một lần chạy trọn vẹn, chưa tính retry khi lỗi.
Nếu hai thứ tự chọn khác nhau, cặp được tính hoà và `position_consistent=false`.
Nếu kết quả không đọc được sau retry, cặp được ghi `failed`, loại khỏi mẫu thống kê và đếm
trong `n_failed`, theo quy tắc NB4.

Tiến độ được lưu sau từng cặp ở `.cache/api-judge-progress.json` (Git bỏ qua). Chạy lại
cùng lệnh để tiếp tục; cặp đã hoàn thành không bị gọi lại. Khi dùng `--limit` mà còn cặp
chưa chấm, script chỉ lưu tiến độ, chưa xuất bộ kết quả cuối. Muốn thử lại các cặp `failed`,
thêm `--retry-failed`; không chấm lại các cặp thành công chỉ để chọn kết quả thuận lợi.
Checkpoint gắn với hash đầu ra, model và prompt chấm; script từ chối trộn các lần chạy
khác cấu hình. Nếu muốn một đối chứng model khác, dùng checkpoint/thư mục kết quả riêng.

## 3. Kết quả và cập nhật bài nộp

Khi đã chấm đủ, script xuất:

- `data/eval/judge_results_api.json`: kết quả từng cặp và quyết định theo hai thứ tự.
- `data/eval/judge_summary_api.json`: win rate, CI bootstrap, position consistency,
  số lỗi và mức đồng thuận với RM.
- `data/eval/judge_summary.json`: giữ nguyên các chỉ số RM, bổ sung `cross_judge`
  và các mức đồng thuận trên held-out / nhóm khác đầu ra.

`cross_judge.agreement` chỉ tính các ID được cả hai bên chấm hợp lệ. Vì có nhiều cặp
giống hệt, cần đọc thêm `cross_judge_heldout_changed`; đồng thuận không đồng nghĩa accuracy.
Giám khảo API không được chạy qua bộ sanity RM, nên `sanity_accuracy=null` trong summary
API, không phải 0 hay 100%.

Sau khi có kết quả thật, bổ sung vào REFLECTION §4: model API, cách chấm hai thứ tự,
win rate/CI theo từng nhóm, position consistency, n_failed, cross_judge và các ví dụ bất đồng.
Đánh dấu bonus chấm chéo; không đổi bảng RM cũ thành bảng API mà không ghi rõ nguồn.
Không suy ra giám khảo nào đúng chỉ vì một bên cho DPO thắng nhiều hơn.

Kiểm tra bài nộp:

```powershell
python -X utf8 scripts/verify.py --training-root /content/lab22
```

Commit kết quả API, mã chạy và REFLECTION đã cập nhật; giữ `.env` và `vlearn.md` ở local.
