mmseqs easy-cluster data/pretrain/pdb_seqres_high_res.txt clusterRes tmp --min-seq-id 0.3 -c 0.8 --cov-mode 1
rm clusterRes_all_seqs.fasta clusterRes_cluster.tsv
rm -rf tmp/
mv clusterRes_rep_seq.fasta data/pretrain/pdb_seqres_0.3.fasta