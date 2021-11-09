export base_path=$1
export epochs=$2

python3 deepcor-noc-full.py --base-path=$base_path --dir=64_nodes_100_ --no-of-epochs=$epochs
python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_100_c
python3 deepcor-noc-full.py --base-path=$base_path --dir=64_nodes_100_c --no-of-epochs=$epochs


python3 deepcor-noc-full.py --base-path=$base_path --dir=64_nodes_95_ --no-of-epochs=$epochs
python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_95_c
python3 deepcor-noc-full.py --base-path=$base_path --dir=64_nodes_95_c --no-of-epochs=$epochs


python3 deepcor-noc-full.py --base-path=$base_path --dir=64_nodes_90_ --no-of-epochs=$epochs
python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_90_c
python3 deepcor-noc-full.py --base-path=$base_path --dir=64_nodes_90_c --no-of-epochs=$epochs


python3 deepcor-noc-full.py --base-path=$base_path --dir=64_nodes_85_ --no-of-epochs=$epochs
python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_85_c
python3 deepcor-noc-full.py --base-path=$base_path --dir=64_nodes_85_c --no-of-epochs=$epochs


# example -: ./run_for_all.sh /export/research26/cyclone/hansika/noc_data 5
