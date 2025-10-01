### Makefile to be included in tests subfolders

# the default fuge-lc executable to use
FUZZY_COCO?=$(TOPLEVEL)/bin/fuzzycoco.exe

# the script to use: must be overriden
SCRIPT=

# the dataset to use: must be overriden
DATASET=
PARAMS=params.json
# the default seed to use
SEED?=123

# the list of available seeds to test: extracted from the content of results/
SEEDS=$(subst .,,$(suffix $(wildcard $(RESULTS)/*)))

# the default output directory
OUTPUT?=.output
OUTPUT_FUZZY_SYSTEM=$(OUTPUT)/fuzzy_system.json
OUTPUT_EVALUATION=$(OUTPUT)/evaluation.json
OUTPUT_PREDICTION=$(OUTPUT)/prediction.csv

# default result dirs
RESULTS=results
RESULTS_FOR_SEED?=$(RESULTS)/seed.$(SEED)
RESULTS_FUZZY_SYSTEM=$(RESULTS_FOR_SEED)/fuzzy_system.json
RESULTS_EVAL=$(RESULTS_FOR_SEED)/evaluation.json
RESULTS_PREDICT=$(RESULTS_FOR_SEED)/prediction.csv

VERBOSE=
VERBOSE_ARG=$(if $(VERBOSE),--verbose,)


tests: test/fuzzy_system/all test/eval/all test/predict/all

test/fuzzy_system: check/available/fuzzy_system compute/fuzzy_system check/fuzzy_system
test/fuzzy_system/all:
	@echo ===== INFERENCE tests =========================================
	@for seed in $(SEEDS); do \
		$(MAKE) test/fuzzy_system SEED=$$seed; \
	done

test/eval: check/eval/available check/eval 
test/eval/all:
	@echo ===== EVALUATION tests =========================================
	for seed in $(SEEDS); do \
		$(MAKE) test/eval SEED=$$seed; \
	done

test/predict: check/predict/available  predict/fuzzy_system check/predict
test/predict/all:
	@echo ===== PREDICTION tests =========================================
	for seed in $(SEEDS); do \
		$(MAKE) test/predict SEED=$$seed; \
	done


profile/fuzzy_system:
	valgrind --tool=callgrind $(FUZZY_COCO) -d $(DATASET) -p $(PARAMS) --seed $(SEED) -o $(OUTPUT_FUZZY_SYSTEM) $(VERBOSE_ARG)
# compute a fuzzy system with $(SEED) and store results in $(OUTPUT)
compute/fuzzy_system:
	$(FUZZY_COCO) -d $(DATASET) -p $(PARAMS) --seed $(SEED) -o $(OUTPUT_FUZZY_SYSTEM) $(VERBOSE_ARG)

check/fuzzy_system: 
	@echo ### testing Fuzzy System inference with SEED=$(SEED)
	@diff $(OUTPUT_FUZZY_SYSTEM) $(RESULTS_FUZZY_SYSTEM) && echo "OK"

check/available/fuzzy_system:
	@test -e $(RESULTS_FUZZY_SYSTEM) || (echo "no fuzzy system results for SEED=$(SEED)"; exit 1)

install/fuzzy_system: clean compute/fuzzy_system
	mkdir -p $(RESULTS_FOR_SEED)
	cp $(OUTPUT_FUZZY_SYSTEM) $(RESULTS_FUZZY_SYSTEM)

eval/fuzzy_system:
	$(FUZZY_COCO) -d $(DATASET) -p $(PARAMS) --seed $(SEED) $(VERBOSE_ARG) -f $(OUTPUT_FUZZY_SYSTEM) --evaluate > $(OUTPUT_EVALUATION)
	@echo "saved evaluation in $(OUTPUT_EVALUATION)"
	@jq  < $(OUTPUT_EVALUATION)

install/eval: eval/fuzzy_system
	mkdir -p $(RESULTS_FOR_SEED)
	cp $(OUTPUT_EVALUATION) $(RESULTS_EVAL)

check/eval/available:
	@test -e $(RESULTS_EVAL) || (echo "no evaluation results for SEED=$(SEED)"; exit 1)

check/eval:
	@echo ### testing Fuzzy System EVALUATION with SEED=$(SEED)
	@diff $(OUTPUT_EVALUATION) $(RESULTS_EVAL) && echo "OK"

predict/fuzzy_system:
	$(FUZZY_COCO) -d $(DATASET) -p $(PARAMS) --seed $(SEED) $(VERBOSE_ARG) -f $(OUTPUT_FUZZY_SYSTEM) --predict > $(OUTPUT_PREDICTION)
	xargs < $(OUTPUT_PREDICTION)

# check that a fuzzy system reference result is available
check/predict/available:
	@test -e $(RESULTS_PREDICT) || (echo "no prediction results for SEED=$(SEED)"; exit 1)

install/predict: predict/fuzzy_system
	mkdir -p $(RESULTS_FOR_SEED)
	cp $(OUTPUT_PREDICTION) $(RESULTS_PREDICT)

# compare the inference check_prediction_available_for_seedof a new fuzzy system vs the reference one
check/predict:
	@echo ### testing Fuzzy System PREDICTION with SEED=$(SEED)
	@diff $(OUTPUT_PREDICTION) $(RESULTS_PREDICT) && echo "OK"


clean:
	rm -rf .output

