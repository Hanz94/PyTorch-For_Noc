export base_path=$1
export epochs=$2

python3 deepcor-test_50.py --base-path=/$base_path --dir=64_nodes_95_c --no-of-epochs=$epochs
python3 deepcor-noc-full_50.py --base-path=$base_path --dir=64_nodes_95_c --no-of-epochs=$epochs


# example -: ./run_for_all.sh /export/research26/cyclone/hansika/noc_data 5
