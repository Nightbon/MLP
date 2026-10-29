BUILD_DIR = ../build

all: install

install:
	cmake -S .. -B $(BUILD_DIR)
	cmake --build $(BUILD_DIR) -j

run: install
	$(BUILD_DIR)/MultilayerPerceptron

tests:
	cmake -S .. -B $(BUILD_DIR) \
		-DBUILD_TESTS=ON

	cmake --build $(BUILD_DIR) \
		--target test_perceptron -j

	$(BUILD_DIR)/test_perceptron

valgrind: tests
	valgrind --tool=memcheck \
		--leak-check=yes \
		--track-origins=yes \
		--show-leak-kinds=all \
		$(BUILD_DIR)/test_perceptron

gcov_report:
	cmake -S .. -B $(BUILD_DIR) \
		-DBUILD_TESTS=ON \
		-DENABLE_GCOV=ON \
		-DCMAKE_BUILD_TYPE=Debug

	cmake --build $(BUILD_DIR) \
		--target test_perceptron -j

	$(BUILD_DIR)/test_perceptron
	mkdir -p $(BUILD_DIR)/report
	gcovr \
	-r . \
	--html \
	--html-details \
	-o $(BUILD_DIR)/report/index.html \
	--filter "model/.*" \
	--exclude "model/model_types.h" \
	--print-summary \
	--exclude-throw-branches \
	--exclude-unreachable-branches \
	--object-directory $(BUILD_DIR)

	xdg-open $(BUILD_DIR)/report/index.html

dvi:
	cmake -S .. -B $(BUILD_DIR)

	cmake --build $(BUILD_DIR) \
		--target dvi

	xdg-open ../docs/html/index.html

dist:
	cd .. && tar czf MLP.tar.gz \
		src \
		CMakeLists.txt \
		README.md \
		README_RUS.md \
		REPORT.md

clean:
	rm -rf $(BUILD_DIR) ../docs/

style:
	find ../src -name "*.cpp" -o -name "*.h" | xargs clang-format -n --style=Google


rebuild: clean install

.PHONY: all install run tests valgrind gcov_report dvi dist clean rebuild style