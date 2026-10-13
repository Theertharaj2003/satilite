@echo off
call .venv\Scripts\activate
python tools\generate_demo_data.py
python scripts\register_pair.py --eo data\raw\eo\demo_eo.png --ir data\raw\ir\demo_ir.png
python scripts\fuse_pair.py --eo data\raw\eo\demo_eo.png --ir data\raw\ir\demo_ir.png
python scripts\run_assistant.py --question "Give me the current scene summary"
