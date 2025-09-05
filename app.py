import os
import json
import re
from collections import Counter
import streamlit as st

from smart_triage import SmartIssueTriage


@st.cache_resource(show_spinner=False)
def load_triage(model_dir: str = 'model_artifacts') -> SmartIssueTriage:
    return SmartIssueTriage(model_dir=model_dir)


def format_tags(pred: dict, min_conf: float = 0.30):
    tags = []
    primary = pred.get('primary_category', {})
    if primary and primary.get('confidence', 0.0) >= min_conf:
        tags.append({
            'tag': primary.get('category', ''),
            'confidence': primary.get('confidence', 0.0)
        })

    for s in pred.get('secondary_suggestions', []):
        if s.get('confidence', 0.0) >= min_conf:
            tags.append({'tag': s.get('category', ''), 'confidence': s.get('confidence', 0.0)})

    # keep only top 3 by confidence
    tags.sort(key=lambda x: x['confidence'], reverse=True)
    return tags[:3]


def main():
    st.set_page_config(page_title='Smart Issue Triage – Tag Suggester', page_icon='🧠', layout='centered')
    st.title('💡 Smart Issue Triage')
    st.subheader('Suggest tags from an issue title and description')

    # Global capsule button styling
    st.markdown(
        """
        <style>
        div.stButton > button, div[data-testid="stButton"] > button {
            border-radius: 999px !important;
            padding: 0.45rem 0.95rem !important;
            border: 1.5px solid rgba(0,0,0,0.15) !important;
            background: rgba(0,0,0,0.04) !important;
            transition: all 120ms ease-in-out;
        }
        .hl { background: #FEF08A; padding: 0.05rem 0.15rem; border-radius: 0.25rem; }
        .hlbox { border: 1px solid rgba(0,0,0,0.08); border-radius: 0.5rem; padding: 0.5rem; background: #fff; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    triage = load_triage()
    repo_options = sorted([str(r) for r in getattr(triage.repo_encoder, 'classes_', [])])
    if not repo_options:
        repo_options = ['unknown_repo']

    # Handle reset BEFORE rendering widgets: clear inputs and state to defaults
    if st.session_state.get('do_reset'):
        st.session_state['input_title'] = ''
        st.session_state['input_body'] = ''
        if repo_options:
            st.session_state['input_repo'] = repo_options[0]
        st.session_state.pop('last_pred', None)
        st.session_state.pop('selected_tags', None)
        st.session_state['do_reset'] = False

    # (Debug sidebar removed)

    with st.form('issue_form'):
        title = st.text_input('Issue Title', '', key='input_title')
        body = st.text_area('Issue Description', height=200, key='input_body')
        repo = st.selectbox('Repository (from training data)', options=repo_options, index=0, key='input_repo')
        submitted = st.form_submit_button('Get Tags')

    # Submit: compute and store prediction, but do not gate rendering on submit thereafter
    if submitted:
        if not title and not body:
            st.warning('Please enter a title or description.')
        else:
            try:
                st.session_state['last_pred'] = triage.predict(title=title, body=body, repo=repo)
            except Exception as e:
                st.error(f'Prediction failed: {e}')

    # Reset form and state for a clean input and prediction
    if st.button('Reset'):
        st.session_state['do_reset'] = True
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()
            except Exception:
                st.stop()

    # Render last prediction suggestions (persist after clicks / reruns)
    pred = st.session_state.get('last_pred')
    if pred:
        tags = pred.get('suggested_tags', [])

        if 'selected_tags' not in st.session_state:
            st.session_state['selected_tags'] = set()

        # Inline highlight preview: top-5 TF-IDF n-grams from current input
        title_val = st.session_state.get('input_title', '')
        body_val = st.session_state.get('input_body', '')
        combined_text = f"{title_val}\n{body_val}"
        top_pairs = []
        try:
            vec = triage.tfidf_vectorizer
            Xv = vec.transform([combined_text])
            weights = Xv.toarray()[0]
            feature_names = getattr(vec, 'get_feature_names_out', vec.get_feature_names)()
            present_idxs = [i for i, w in enumerate(weights) if w > 0]
            top_idxs = sorted(present_idxs, key=lambda i: weights[i], reverse=True)[:5]
            top_tokens = [feature_names[i] for i in top_idxs]
            top_pairs = [(feature_names[i], float(weights[i])) for i in top_idxs]

            # Fallback: if no TF-IDF overlap, pick top 5 tokens by frequency using vectorizer analyzer
            if not top_tokens:
                analyzer = vec.build_analyzer()
                toks = [t for t in analyzer(combined_text) if len(t) >= 3]
                counts = Counter(toks)
                most_common = [tok for tok, _ in counts.most_common(5)]
                top_tokens = most_common
                top_pairs = [(tok, float(counts[tok])) for tok in most_common]

            def highlight_html(text: str, tokens):
                html = text
                for tok in sorted(tokens, key=lambda t: len(t), reverse=True):
                    if not tok:
                        continue
                    pattern = re.compile(re.escape(tok), flags=re.IGNORECASE)
                    html = pattern.sub(lambda m: f"<span class='hl'>{m.group(0)}</span>", html)
                return html

            if any(top_tokens):
                st.caption('Top terms highlighted in your input')
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('Title preview:')
                    st.markdown(f"<div class='hlbox'>{highlight_html(title_val, top_tokens)}</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown('Description preview:')
                    st.markdown(f"<div class='hlbox' style='min-height: 6rem;'>{highlight_html(body_val, top_tokens)}</div>", unsafe_allow_html=True)
        except Exception:
            pass

        st.success('Suggested tags:')
        cols = st.columns(len(tags) if tags else 1)
        # Add per-column button opacity styling using nth-child targeting
        for i, t in enumerate(tags):
            tag = t.get('tag', '')
            conf = float(t.get('confidence', 0))
            opacity = max(0.25, min(1.0, conf))
            is_selected = tag in st.session_state['selected_tags']

            # Target the i-th column's button to set opacity and selected border
            st.markdown(
                f"""
                <style>
                div[data-testid="column"]:nth-child({i+1}) button {{
                    opacity: {1.0 if is_selected else opacity} !important;
                    border-color: {'#16a34a' if is_selected else 'rgba(0,0,0,0.15)'} !important;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )

            with cols[i]:
                label = f"{tag} — {conf*100:.0f}%"
                if st.button(label if not is_selected else f"✅ {label}", key=f"tag_btn_{i}"):
                    if is_selected:
                        st.session_state['selected_tags'].discard(tag)
                    else:
                        st.session_state['selected_tags'].add(tag)

        with st.expander('Details'):
            st.markdown('Top 5 terms influencing suggestions (TF-IDF weight or frequency fallback):')
            if top_pairs:
                for tok, w in top_pairs:
                    st.write(f"- {tok} — {w:.3f}")
            else:
                st.info('No significant terms found in this input.')


if __name__ == '__main__':
    main()


