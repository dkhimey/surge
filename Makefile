# === Compilers ===
MPICXX := mpic++
GXX := g++

# === Flags ===
CXXFLAGS := -O3 -std=c++17 -fopenmp -Wall -Iinclude -Wno-int-in-bool-context
LDFLAGS := -fopenmp

# === Include Paths ===
INCLUDES := -Iexternal/hnswlib/hnswlib \
            -Iexternal/KaHIP/interface \
            -Iexternal/nlohmann/ \
            -Iexternal/yaml-cpp/include

# === KaHIP Settings ===
KAHIP_DIR := external/KaHIP
KAHIP_LIB := $(KAHIP_DIR)/lib/libKaHIP.a

# === YAML Settings ===
YAML_CPP_DIR := external/yaml-cpp
YAML_CPP_LIB := $(YAML_CPP_DIR)/build/libyaml-cpp.a

# === Directories ===
SRC_DIR := src
OBJ_DIR := build

# === Core ===
CORE_SRCS := $(SRC_DIR)/index.cpp $(SRC_DIR)/utils.cpp
CORE_OBJS := $(CORE_SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# === Experiments ===
# Top-level drivers plus any grouped into sub-directories (e.g. theoretical_dynamic_simulation).
# Binaries are always emitted flat as bin/<basename>.
EXP_DIR := experiments
EXP_SRCS := $(wildcard $(EXP_DIR)/*.cpp) $(wildcard $(EXP_DIR)/*/*.cpp)
EXP_BINS := $(addprefix bin/,$(notdir $(EXP_SRCS:.cpp=)))

# Resolve experiment sources by basename regardless of sub-directory.
vpath %.cpp $(sort $(dir $(EXP_SRCS)))

# === Old Experiments ===
OLD_DIR := old_experiments
OLD_SRCS := $(wildcard $(OLD_DIR)/*.cpp)
OLD_OBJS := $(OLD_SRCS:$(OLD_DIR)/%.cpp=$(OBJ_DIR)/old_%.o)
OLD_BINS := $(OLD_SRCS:$(OLD_DIR)/%.cpp=bin/old_%)

# === Default: core only ===
all: $(CORE_OBJS)

# === Experiment targets ===
experiments: $(EXP_BINS)

old_experiments: $(OLD_BINS)

# === Compile rules ===
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/exp_%.o: %.cpp
	@mkdir -p $(dir $@)
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(OBJ_DIR)/old_%.o: $(OLD_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(MPICXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# === Link rules ===
bin/%: $(OBJ_DIR)/exp_%.o $(CORE_OBJS) $(KAHIP_LIB) $(YAML_CPP_LIB)
	@mkdir -p bin
	$(MPICXX) $(LDFLAGS) $^ -o $@ -L$(KAHIP_DIR)/lib -lKaHIP

bin/old_%: $(OBJ_DIR)/old_%.o $(CORE_OBJS) $(KAHIP_LIB) $(YAML_CPP_LIB)
	@mkdir -p bin
	$(MPICXX) $(LDFLAGS) $^ -o $@ -L$(KAHIP_DIR)/lib -lKaHIP

# === KaHIP Build ===
$(KAHIP_LIB):
	cd $(KAHIP_DIR) && mkdir -p build && cd build && cmake .. -DNOMPI=ON && make
	cp $(KAHIP_DIR)/build/libkahip_static.a $(KAHIP_LIB)

# === YAML Build ===
$(YAML_CPP_LIB):
	cd $(YAML_CPP_DIR) && mkdir -p build && cd build && cmake .. -DBUILD_SHARED_LIBS=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON && make

# === Clean ===
clean:
	rm -rf build bin
	rm -rf $(KAHIP_DIR)/build $(KAHIP_LIB)
	rm -rf $(YAML_CPP_DIR)/build

.PHONY: all clean experiments old_experiments