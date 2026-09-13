import time
import pandas as pd
import streamlit as st

from services.config import get_config
from services.pdf_service import extract_pdf_text, extract_pages_with_text, get_pdf_page_count
from services.groq_service import grade_student_paper, vision_ocr_pdf
from services.analytics import results_to_dataframe, build_question_analytics, detect_review_flags
from services.exports import make_excel, make_json_zip

st.set_page_config(page_title="GradeFlow AI", page_icon="📝", layout="wide")
cfg = get_config()

if "results" not in st.session_state:
    st.session_state.results = []

st.title("📝 GradeFlow AI")
st.caption("AI-assisted exam checking • 1 PDF = 1 student • each student PDF can contain 2+ pages")

with st.sidebar:
    st.header("⚙️ Settings")
    grading_model = st.text_input("Grading model", value=cfg.grading_model)
    vision_model = st.text_input("Vision/OCR model", value=cfg.vision_model)
    max_marks = st.number_input("Exam maximum marks", min_value=1, max_value=10000, value=100)
    review_threshold = st.slider("Manual review threshold", 0.0, 1.0, cfg.review_threshold, 0.05)
    use_vision = st.checkbox("Use Vision OCR for scanned/handwritten PDFs", value=True)
    max_pages = st.number_input("Maximum pages per student PDF", min_value=1, max_value=100, value=max(20, cfg.max_vision_pages))
    st.divider()
    if cfg.api_key:
        st.success("Groq API key loaded")
    else:
        st.error("GROQ_API_KEY missing")
    st.caption("Rate-safe mode: digital pages use local extraction; scanned pages use serialized Vision OCR; grading has automatic JSON fallback and 429 retry.")

st.markdown("## 1. Exam setup")
c1, c2 = st.columns(2)
with c1:
    question_file = st.file_uploader("Question paper PDF", type=["pdf"], key="question_file")
with c2:
    answer_key_file = st.file_uploader("Official answer key PDF", type=["pdf"], key="answer_key_file")

question_text = ""
answer_key_text = ""

if question_file and answer_key_file:
    question_text = extract_pdf_text(question_file.getvalue())
    answer_key_text = extract_pdf_text(answer_key_file.getvalue())

    if use_vision and len(question_text.strip()) < cfg.min_text_chars_for_ocr:
        with st.status("Reading scanned question paper with Vision...", expanded=False):
            question_text = vision_ocr_pdf(cfg.api_key, question_file.getvalue(), vision_model, int(max_pages))

    if use_vision and len(answer_key_text.strip()) < cfg.min_text_chars_for_ocr:
        with st.status("Reading scanned answer key with Vision...", expanded=False):
            answer_key_text = vision_ocr_pdf(cfg.api_key, answer_key_file.getvalue(), vision_model, int(max_pages))

    a, b = st.columns(2)
    a.metric("Question paper characters", f"{len(question_text):,}")
    b.metric("Answer key characters", f"{len(answer_key_text):,}")

st.markdown("## 2. Upload student PDFs")
st.info("Upload one PDF per student. Each PDF may contain 2, 3, 5, 10 or more pages. All pages inside one PDF are treated as ONE student's submission.")
student_files = st.file_uploader(
    "Student answer-sheet PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    key="student_files",
)

if student_files:
    total_mb = sum(len(f.getvalue()) for f in student_files) / (1024 * 1024)
    st.success(f"✅ {len(student_files)} student PDF(s) selected • {total_mb:.1f} MB total")
    with st.expander("Preview uploaded papers"):
        preview = []
        for f in sorted(student_files, key=lambda x: x.name.lower()):
            data = f.getvalue()
            preview.append({"File": f.name, "Pages": get_pdf_page_count(data), "Size (MB)": round(len(data) / (1024 * 1024), 2)})
        st.dataframe(pd.DataFrame(preview), use_container_width=True, hide_index=True)

st.markdown("## 3. Run batch")

if st.button("🚀 Start bulk grading", type="primary", use_container_width=True):
    if not cfg.api_key:
        st.error("GROQ_API_KEY is missing. Add it in Streamlit Cloud → Settings → Secrets.")
        st.stop()
    if not question_file or not answer_key_file:
        st.error("Upload the question paper and official answer key first.")
        st.stop()
    if not question_text.strip() or not answer_key_text.strip():
        st.error("The question paper or answer key could not be read. Enable Vision OCR and try again.")
        st.stop()
    if not student_files:
        st.error("Upload at least one student PDF.")
        st.stop()

    ordered_files = sorted(student_files, key=lambda x: x.name.lower())
    total = len(ordered_files)
    overall = st.progress(0, text="Starting batch...")
    headline = st.empty()
    detail = st.empty()
    result_placeholder = st.empty()
    results = []
    started = time.time()

    for index, uploaded in enumerate(ordered_files, start=1):
        pdf_bytes = uploaded.getvalue()
        page_count = get_pdf_page_count(pdf_bytes)
        headline.info(f"📄 Processing student {index} of {total}: **{uploaded.name}** ({page_count} page(s))")
        detail.write("🔎 Step 1/3 — Reading PDF and identifying pages...")

        try:
            page_info = extract_pages_with_text(pdf_bytes, min_chars=25)
            selected_pages = [p for p in page_info if p["page"] <= int(max_pages)]
            scanned_pages = [p["page"] for p in selected_pages if p["needs_ocr"]]
            digital_pages = len(selected_pages) - len(scanned_pages)

            detail.write(f"🔎 Step 1/3 — {digital_pages} digital page(s) + {len(scanned_pages)} scanned/handwritten page(s) detected.")

            if use_vision:
                detail.write("👁️ Step 2/3 — Reading all pages; Vision is used only where local text extraction is insufficient...")
                student_text = vision_ocr_pdf(cfg.api_key, pdf_bytes, vision_model, int(max_pages))
            else:
                student_text = extract_pdf_text(pdf_bytes)
                if not student_text.strip():
                    raise ValueError("No selectable text found. Enable Vision OCR for scanned/handwritten papers.")

            if not student_text.strip():
                raise ValueError("No readable text could be extracted from this PDF.")

            detail.write("🤖 Step 3/3 — Grading the complete multi-page student submission...")
            result = grade_student_paper(
                api_key=cfg.api_key,
                grading_model=grading_model,
                question_paper=question_text,
                answer_key=answer_key_text,
                student_text=student_text,
                max_marks=int(max_marks),
            )
            result["filename"] = uploaded.name
            result["page_count"] = page_count
            result["review_required"] = (
                float(result.get("confidence", 0)) < review_threshold
                or result.get("student_name", "Unknown") in ["Unknown", ""]
                or result.get("roll_no", "Unknown") in ["Unknown", ""]
            )
            result["review_reasons"] = detect_review_flags(result, review_threshold)
            results.append(result)
            detail.success(f"✅ Completed {index}/{total}: {result.get('student_name', 'Unknown')} • {result.get('roll_no', 'Unknown')} • {result.get('score', 0)}/{result.get('total_marks', max_marks)}")

        except Exception as exc:
            error_result = {
                "filename": uploaded.name,
                "page_count": page_count,
                "student_name": "ERROR",
                "roll_no": "ERROR",
                "score": 0,
                "total_marks": int(max_marks),
                "percentage": 0,
                "grade": "ERROR",
                "confidence": 0,
                "feedback": str(exc),
                "question_results": [],
                "review_required": True,
                "review_reasons": [str(exc)],
                "error": str(exc),
            }
            results.append(error_result)
            detail.error(f"⚠️ {uploaded.name} failed, but the batch continues: {exc}")

        overall.progress(index / total, text=f"Overall progress: {index}/{total} students")
        partial_df = results_to_dataframe(results)
        result_placeholder.dataframe(partial_df, use_container_width=True, hide_index=True)

    elapsed = time.time() - started
    st.session_state.results = results
    headline.success(f"🎉 Batch complete — {len(results)} student(s) processed in {elapsed / 60:.1f} minutes.")
    detail.write("You can now review the consolidated result sheet and export Excel.")

if st.session_state.results:
    df = results_to_dataframe(st.session_state.results)
    st.markdown("---")
    st.markdown("## 📊 Consolidated result sheet")
    valid = pd.to_numeric(df["Score"], errors="coerce")
    review_count = int(df["Review Required"].fillna(False).sum())
    error_count = int((df["Status"] == "ERROR").sum())
    a, b, c, d, e = st.columns(5)
    a.metric("Students", len(df))
    b.metric("Average", f"{valid.mean():.1f}" if valid.notna().any() else "0.0")
    c.metric("Highest", f"{valid.max():.1f}" if valid.notna().any() else "0.0")
    d.metric("Review", review_count)
    e.metric("Errors", error_count)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("## 📈 Class analytics")
    qdf = build_question_analytics(st.session_state.results)
    if not qdf.empty:
        st.bar_chart(qdf.set_index("Question")["Average Marks"])
        st.dataframe(qdf, use_container_width=True, hide_index=True)
    st.markdown("## 📝 Detailed question grading")

    for r in st.session_state.results:
        with st.expander(
            f"{r.get('student_name', 'Unknown')} • "
            f"{r.get('roll_no', 'Unknown')} • "
            f"{r.get('score', 0)}/{r.get('total_marks', max_marks)}"
        ):
                    st.markdown(f"### Question {question}")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Student Answer**")
            st.write(qr.get("student_answer", "Not available"))

        with col2:
            st.markdown("**Correct Answer**")
            st.write(qr.get("correct_answer", "Not available"))

        with col3:
            st.markdown("**AI Reason**")
            st.write(qr.get("reason", "No reason provided."))

                    st.markdown(
                        f"### Question {question} — "
                        f"{marks}/{max_q_marks} marks ({status})"
                    )

                    c1, c2, c3 = st.columns(3)

                    with c1:
                        st.markdown("**Student Answer**")
                        st.write(qr.get("student_answer", "Not available"))

                    with c2:
                        st.markdown("**Correct Answer**")
                        st.write(qr.get("correct_answer", "Not available"))

                    with c3:
                        st.markdown("**AI Reason**")
                        st.write(qr.get("reason", "No reason provided."))
        teacher_marks = st.number_input(
            f"Teacher marks — Question {question}",
            min_value=0.0,
            max_value=max_q_marks,
            value=ai_marks,
            step=0.5,
            key=f"teacher_marks_{r_index}_{q_index}"
        )

        qr["teacher_marks"] = teacher_marks

                    st.divider()
            else:
                st.info("No question-level results available.")

    st.markdown("## ⚠️ Teacher review queue")

review_items = [
    r for r in st.session_state.results
    if r.get("review_required")
]

if not review_items:
    st.success("No papers currently require manual review.")
else:
    st.warning(f"{len(review_items)} paper(s) require review.")

    for r_index, r in enumerate(review_items):
        with st.expander(
            f"{r.get('student_name')} • "
            f"{r.get('roll_no')} • "
            f"{r.get('score')}/{r.get('total_marks')}"
        ):
            for reason in r.get("review_reasons", []):
                st.write(f"- {reason}")

            st.write(r.get("feedback", ""))

            st.markdown("### ✏️ Teacher correction")

            if r.get("question_results"):
                for q_index, qr in enumerate(r["question_results"]):
                    question = qr.get("question", "Unknown")
                    ai_marks = float(qr.get("marks_awarded", 0))
                    max_q_marks = float(qr.get("max_marks", 0))

                    teacher_marks = st.number_input(
                        f"Question {question} marks",
                        min_value=0.0,
                        max_value=max_q_marks,
                        value=ai_marks,
                        step=0.5,
                        key=f"teacher_marks_{r_index}_{q_index}"
                    )

                    qr["teacher_marks"] = teacher_marks

            if st.button(
                "✅ Finalize teacher corrections",
                key=f"finalize_{r_index}"
            ):
                total_score = 0.0

                for qr in r["question_results"]:
                    final_marks = qr.get(
                        "teacher_marks",
                        qr.get("marks_awarded", 0)
                    )
                    qr["marks_awarded"] = final_marks
                    total_score += float(final_marks)

                r["score"] = total_score
                r["total_marks"] = sum(
                    float(qr.get("max_marks", 0))
                    for qr in r["question_results"]
                )

                if r["total_marks"] > 0:
                    r["percentage"] = (
                        total_score / r["total_marks"]
                    ) * 100

                r["teacher_finalized"] = True
                r["review_required"] = False

                st.success(
                    f"Finalized! Final score: "
                    f"{r['score']}/{r['total_marks']}"
                )
                st.rerun()

    st.markdown("## ⬇️ Export")
    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download Excel result sheet",
            data=make_excel(st.session_state.results),
            file_name="gradeflow_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "Download JSON feedback ZIP",
            data=make_json_zip(st.session_state.results),
            file_name="gradeflow_feedback.zip",
            mime="application/zip",
            use_container_width=True,
        )
