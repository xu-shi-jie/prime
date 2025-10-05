./scripts/pretrain/1.download.sh
python scripts/pretrain/2.remove_low_res.py
./scripts/pretrain/3.rm_redundancy.sh
python scripts/pretrain/4.gen_ds.py