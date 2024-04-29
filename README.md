# Flare Remover in Web Page

## pretrained weight
- [Depth model](https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt)
- [Flare_removal](https://drive.google.com/file/d/1iyyVimG1mOckJFu4faq9sCo0Bnk65Yc7/view?usp=drive_link)

## Usage
1. Run streamlights at the same time you specify a port
   ```
   streamlit run home.py --server.port=80
   ```
2. Converting Local to Global with ngrok
   ```
   ngrok http 80
   ```
