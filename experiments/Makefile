# Offline estimator evaluation harness -- self-contained.
#
# Depends only on Eigen and the estimator headers under ../src (pulled in by the
# relative includes in main.cpp). It does NOT link the streaming server, so this
# folder can be gitignored or split into its own repo. Run from THIS directory:
#
#   make && ./experiment jump        # writes results/experiment_jump.csv
#   for s in constant drift jump stochastic oscillate; do ./experiment $s; done

CXX = clang++
EIGEN = /opt/homebrew/opt/eigen

experiment:
	$(CXX) main.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o experiment

calibrate_arl0:
	$(CXX) calibrate_arl0.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o calibrate_arl0

multiseed:
	$(CXX) multiseed.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o multiseed

probe_rank_score:
	$(CXX) probe_rank_score.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o probe_rank_score

calibrate_tail:
	$(CXX) calibrate_tail.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o calibrate_tail

probe_tail:
	$(CXX) probe_tail.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o probe_tail

probe_bank:
	$(CXX) probe_bank.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o probe_bank

probe_family:
	$(CXX) probe_family.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o probe_family

probe_auc:
	$(CXX) probe_auc.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o probe_auc

probe_shape:
	$(CXX) probe_shape.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o probe_shape

probe_score_scale:
	$(CXX) probe_score_scale.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o probe_score_scale

probe_xiao_pit:
	$(CXX) probe_xiao_pit.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o probe_xiao_pit

probe_memory:
	$(CXX) probe_memory.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o probe_memory

probe_adaptive:
	$(CXX) probe_adaptive.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o probe_adaptive

probe_vff_norm:
	$(CXX) probe_vff_norm.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o probe_vff_norm

probe_memory_tradeoff:
	$(CXX) probe_memory_tradeoff.cpp -std=c++20 -O2 -o probe_memory_tradeoff

dump_copula_samples:
	$(CXX) dump_copula_samples.cpp -std=c++20 -O2 -o dump_copula_samples

tail_validate:
	$(CXX) tail_validate.cpp -std=c++20 -O2 -pthread -I$(EIGEN)/include/eigen3 -o tail_validate

tail_change:
	$(CXX) tail_change.cpp -std=c++20 -O2 -pthread -I$(EIGEN)/include/eigen3 -o tail_change

tail_mixture:
	$(CXX) tail_mixture.cpp -std=c++20 -O2 -pthread -I$(EIGEN)/include/eigen3 -o tail_mixture

tail_soft:
	$(CXX) tail_soft.cpp -std=c++20 -O2 -pthread -I$(EIGEN)/include/eigen3 -o tail_soft

async_memory:
	$(CXX) async_memory.cpp -std=c++20 -O2 -pthread -I$(EIGEN)/include/eigen3 -o async_memory

async_noise:
	$(CXX) async_noise.cpp -std=c++20 -O2 -pthread -I$(EIGEN)/include/eigen3 -o async_noise

clean:
	rm -f experiment calibrate_arl0 multiseed probe_auc probe_score_scale probe_shape probe_rank_score probe_family probe_bank probe_xiao_pit probe_tail calibrate_tail probe_memory probe_adaptive probe_vff_norm probe_memory_tradeoff dump_copula_samples tail_validate tail_change tail_mixture tail_soft async_memory async_noise

.PHONY: experiment calibrate_arl0 multiseed probe_auc probe_score_scale probe_shape probe_rank_score probe_family probe_bank probe_xiao_pit probe_tail calibrate_tail probe_memory probe_adaptive probe_vff_norm probe_memory_tradeoff dump_copula_samples tail_validate tail_change tail_mixture tail_soft async_memory async_noise clean

# The other direction: a make_estimator branch with no all_estimator_keys entry
# runs never and reports nothing. Compares the two lists in registry.hpp.
check-registry:
	@python3 -c "import re,sys; \
s=open('../src/estimators/registry.hpp').read(); \
br=set(re.findall(r'key == \"([a-z0-9_]+)\"', s)); \
ls=set(re.findall(r'\"([a-z0-9_]+)\"', re.search(r'all_estimator_keys.*?\{(.*?)\};', s, re.S).group(1))); \
a=sorted(br-ls); b=sorted(ls-br); \
print('branch but unlisted (never runs):', a or 'none'); \
print('listed but no branch (segfaults):', b or 'none'); \
sys.exit(1 if a or b else 0)"

bench_rate:
	$(CXX) bench_rate.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o bench_rate

check_thesis_bank:
	$(CXX) check_thesis_bank.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o check_thesis_bank

probe_thesis_bank:
	$(CXX) probe_thesis_bank.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o probe_thesis_bank

probe_bank_tune:
	$(CXX) probe_bank_tune.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o probe_bank_tune

probe_bank_path:
	$(CXX) probe_bank_path.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o probe_bank_path

probe_family_bank:
	$(CXX) probe_family_bank.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o probe_family_bank

probe_tail_bank:
	$(CXX) probe_tail_bank.cpp -std=c++20 -O2 -pthread -I$(EIGEN)/include/eigen3 -o probe_tail_bank

probe_tail_ci:
	$(CXX) probe_tail_ci.cpp -std=c++20 -O2 -pthread -I$(EIGEN)/include/eigen3 -o probe_tail_ci

probe_batch_ise:
	$(CXX) probe_batch_ise.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o probe_batch_ise

probe_family_path:
	$(CXX) probe_family_path.cpp -std=c++20 -O2 -pthread -I$(EIGEN)/include/eigen3 -o probe_family_path

probe_bank_regret:
	$(CXX) probe_bank_regret.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o probe_bank_regret
