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

probe_family:
	$(CXX) probe_family.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o probe_family

probe_shape:
	$(CXX) probe_shape.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o probe_shape

probe_score_scale:
	$(CXX) probe_score_scale.cpp -std=c++20 -O2 -I$(EIGEN)/include/eigen3 -o probe_score_scale

clean:
	rm -f experiment calibrate_arl0 multiseed probe_score_scale probe_shape probe_rank_score probe_family

.PHONY: experiment calibrate_arl0 multiseed probe_score_scale probe_shape probe_rank_score probe_family clean

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
