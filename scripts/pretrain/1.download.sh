rm data/pretrain/pdb_seqres.txt
aria2c -x 16 https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz -d data/pretrain/
gunzip data/pretrain/pdb_seqres.txt.gz