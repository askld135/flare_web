# Flare Remover in Web Page

## pretrained weight
- [Depth model](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt)
- [Flare_removal](https://drive.google.com/file/d/1iyyVimG1mOckJFu4faq9sCo0Bnk65Yc7/view?usp=drive_link)
- [Instance_Segmentation](https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet-ins_x_8xb16-300e_coco/rtmdet-ins_x_8xb16-300e_coco_20221124_111313-33d4595b.pth)

## Usage
1. Run streamlights at the same time you specify a port
   ```
   streamlit run home.py --server.port=80
   ```
   and Change the name of the page file to pages.
   if you want to run home1.py
   ```
   streamlit run home1.py --server.port=80
   ```
   and Change the name of the pages file to page.
   
2. Converting Local to Global with ngrok
   ```
   ngrok http 80
   ```
