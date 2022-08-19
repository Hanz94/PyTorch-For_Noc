export base_path=$1
export epochs=$2

python3 deepcor-train-n.py --base-path=$base_path --dir=64_nodes_85__0.01_f_150 --no-of-epochs=$epochs --no-of-flits=150
python3 deepcor-train-n.py --base-path=$base_path --dir=64_nodes_85__0.01_f_250 --no-of-epochs=$epochs --no-of-flits=250

python3 deepcor-train-n.py --base-path=$base_path --dir=64_nodes_85__0.01_f_350 --no-of-epochs=$epochs --no-of-flits=350
python3 deepcor-train-n.py --base-path=$base_path --dir=64_nodes_85__0.01_f_450 --no-of-epochs=$epochs --no-of-flits=450

python3 deepcor-train-n.py --base-path=$base_path --dir=64_nodes_85__0.01_f_550 --no-of-epochs=$epochs --no-of-flits=550
python3 deepcor-train-n.py --base-path=$base_path --dir=64_nodes_85__0.01_f_650 --no-of-epochs=$epochs --no-of-flits=650


# python3 deepcor-train-n.py --base-path=$base_path --dir=64_nodes_100_ --no-of-epochs=$epochs
# python3 deepcor-train-n.py --base-path=$base_path --dir=64_nodes_95_ --no-of-epochs=$epochs
# python3 deepcor-train-n.py --base-path=$base_path --dir=64_nodes_90_ --no-of-epochs=$epochs

# python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_100_c --no-of-epochs=$epochs
# python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_95_c --no-of-epochs=$epochs
# python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_90_c --no-of-epochs=$epochs

# python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_100_c --no-of-epochs=$epochs
# python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_95_c --no-of-epochs=$epochs
# python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_90_c --no-of-epochs=$epochs

# python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_100_d --no-of-epochs=$epochs
# python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_95_d --no-of-epochs=$epochs
# python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_90_d --no-of-epochs=$epochs

# python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_100_d --no-of-epochs=$epochs
# python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_95_d --no-of-epochs=$epochs
# python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_90_d --no-of-epochs=$epochs

# python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_100_cd --no-of-epochs=$epochs
# python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_95_cd --no-of-epochs=$epochs
# python3 deepcor-train-o.py --base-path=$base_path --dir=64_nodes_90_cd --no-of-epochs=$epochs

# python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_100_cd --no-of-epochs=$epochs
# python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_95_cd --no-of-epochs=$epochs
# python3 deepcor-test.py --base-path=/$base_path --dir=64_nodes_90_cd --no-of-epochs=$epochs



# example -: ./run_for_all.sh /export/research26/cyclone/hansika/noc_data 5
