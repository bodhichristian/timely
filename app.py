import os
import json
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
    st.title('🧠 Smart Issue Triage')
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
        </style>
        """,
        unsafe_allow_html=True,
    )

    triage = load_triage()
    repo_options = sorted([str(r) for r in getattr(triage.repo_encoder, 'classes_', [])])
    if not repo_options:
        repo_options = ['unknown_repo']

    # Handle reset BEFORE rendering widgets
    if st.session_state.get('do_reset'):
        for k in ['input_title', 'input_body', 'input_repo', 'last_pred', 'selected_tags']:
            st.session_state.pop(k, None)
        st.session_state['do_reset'] = False

    # Debug section (UI only, not wired to results yet)
    st.sidebar.header('Debug')
    st.sidebar.caption('Developer controls (not applied to results)')
    _dbg_threshold = st.sidebar.slider(
        'Confidence threshold (debug)', min_value=0.0, max_value=1.0, value=0.30, step=0.05
    )

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
        st.experimental_rerun()

    # Render last prediction suggestions (persist after clicks / reruns)
    pred = st.session_state.get('last_pred')
    if pred:
        tags = pred.get('suggested_tags', [])

        if 'selected_tags' not in st.session_state:
            st.session_state['selected_tags'] = set()

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
            st.json(pred)


if __name__ == '__main__':
    main()


