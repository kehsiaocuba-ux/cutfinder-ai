import cv2
import streamlit as st
from pathlib import Path
import tempfile
import json

def detect_cuts(video_path, threshold=30, min_scene_len=15):
    """Enhanced detection with minimum scene length."""
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    cuts = []
    frame_count = 0
    last_cut_frame = 0
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if prev_frame is None:
            prev_frame = gray
            continue
            
        frame_delta = cv2.absdiff(prev_frame, gray)
        diff_score = frame_delta.mean()
        
        # Only register cut if minimum scene length passed
        if diff_score > threshold and (frame_count - last_cut_frame) > min_scene_len:
            timecode = frame_count / fps
            cuts.append({
                "frame": frame_count,
                "timecode": f"{int(timecode//3600):02d}:{int((timecode%3600)//60):02d}:{int(timecode%60):02d}:{int((timecode*fps)%fps):02d}"
            })
            last_cut_frame = frame_count
            
        prev_frame = gray
        frame_count += 1
        
    cap.release()
    return cuts, fps

def generate_edl(cuts, fps, video_name="CLIP"):
    """Generate CMX 3600 EDL format."""
    edl_lines = ["TITLE: CutFinder AI EDL", "FCM: NON-DROP FRAME"]
    for i, cut in enumerate(cuts):
        # Simplified EDL entry
        edl_lines.append(f"{i+1:03d}  AX       V     C        {cut['timecode']} {cut['timecode']} {cut['timecode']} {cut['timecode']}")
    return "\n".join(edl_lines)

st.title("CutFinder AI")
st.subheader("Auto-Detect Edit Points - Professional EDL Export")

uploaded_file = st.file_uploader("Upload video (Free: 100MB limit)", type=['mp4', 'mov', 'avi'])

if uploaded_file:
    if uploaded_file.size > 100e6:
        st.error("🚫 Free version limited to 100MB")
        st.info("💰 **Go Pro for unlimited processing + EDL export**")
        if st.button("Get CutFinder Pro - \$29"):
            t.markdown("[Buy on Gumroad](https://kehsiaocube.gumroad.com/l/iufrqd)", unsafe_allow_html=True)
    else:
        with st.spinner("Analyzing video..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                tmp.write(uploaded_file.read())
                cuts, fps = detect_cuts(tmp.name)
                
        st.success(f"✅ Found **{len(cuts)}** cut points")
        
        # Show results
        st.json(cuts)
        
        # EDL download (Pro teaser)
        if len(cuts) > 5:
            st.warning("🔒 EDL export limited to 5 cuts in free version")
            cuts_for_edl = cuts[:5]
        else:
            cuts_for_edl = cuts
            
        edl_content = generate_edl(cuts_for_edl, fps)
        st.download_button(
            label="Download EDL (Pro for full export)",
            data=edl_content,
            file_name="cuts.edl",
            mime="text/plain"

        )
