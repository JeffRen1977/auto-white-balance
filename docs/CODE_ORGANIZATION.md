# Code Organization Guide

This document explains how the codebase is organized for easy understanding and learning.

## 📁 Directory Structure

**Important:** The virtual environment should be located one directory above this project:

```
Mobile_Computational_photograph/
├── virtual_env/                 # Virtual environment (create here)
│   ├── bin/
│   ├── lib/
│   └── ...
└── AutoWhiteBalance/            # This project
    │
    ├── README.md                # Main entry point - start here!
    │
    ├── src/                     # Core Implementation
    │   └── awb_system.py       # Complete AWB system (production code)
    │
    ├── examples/                # Example Scripts
    │   ├── basic_usage.py      # Simple AWB example (beginner-friendly)
    │   ├── run.sh              # Easy execution script (auto-activates venv)
    │   └── setup_venv.sh       # Virtual environment setup script
    │
    ├── tests/                   # Test Scripts
    │   └── test_real_data.py   # Real RAW sensor data testing
    │
    ├── data/                    # Test Data
    │   └── *.raw               # RAW10 sensor data files
    │
    ├── docs/                    # Documentation
    │   ├── INTRODUCTION.md     # What is white balance? (concepts)
    │   ├── GETTING_STARTED.md  # Quick start guide
    │   ├── TUTORIAL.md         # Step-by-step tutorial
    │   ├── ALGORITHMS.md       # Algorithm explanations
    │   ├── API_REFERENCE.md    # Complete API documentation
    │   ├── CODE_ORGANIZATION.md # This file
    │   ├── USAGE.md            # Usage examples
    │   └── TEST_RESULTS.md     # Test results and analysis
    │
    ├── results/                 # Output Results
    │   ├── basic_example/      # Basic example outputs
    │   └── awb_real_data_results/ # Real data test outputs
    │
    └── requirements.txt         # Python dependencies
```

**Virtual Environment Setup:**
```bash
# Create virtual environment in parent directory
cd ..
python3 -m venv virtual_env

# Or use the setup script
cd AutoWhiteBalance
./examples/setup_venv.sh
```

## 🎯 Where to Start

### For Complete Beginners

1. **Read**: `README.md` - Overview and quick start
2. **Read**: `docs/INTRODUCTION.md` - Understand white balance concepts
3. **Run**: `examples/basic_usage.py` - See AWB in action
4. **Read**: `docs/TUTORIAL.md` - Step-by-step implementation

### For Those with Some Experience

1. **Read**: `docs/GETTING_STARTED.md` - Quick setup
2. **Study**: `src/awb_system.py` - Production implementation
3. **Test**: `tests/test_real_data.py` - Real data validation
4. **Read**: `docs/ALGORITHMS.md` - Algorithm details

### For Advanced Users

1. **Study**: `src/awb_system.py` - Complete implementation
2. **Modify**: Create custom analyzers
3. **Integrate**: Use in your camera system
4. **Reference**: `docs/API_REFERENCE.md` - Complete API

## 📚 Documentation Files

### Core Documentation

| File | Purpose | Audience |
|------|---------|----------|
| `README.md` | Main overview, quick start | Everyone |
| `docs/INTRODUCTION.md` | White balance concepts | Beginners |
| `docs/GETTING_STARTED.md` | Installation and first steps | Beginners |
| `docs/TUTORIAL.md` | Step-by-step implementation | Learners |
| `docs/ALGORITHMS.md` | Algorithm deep dive | Intermediate |
| `docs/API_REFERENCE.md` | Complete API documentation | Developers |
| `docs/USAGE.md` | Usage examples | All levels |
| `docs/TEST_RESULTS.md` | Test results and analysis | All levels |

### Reading Order

**Beginner Path**:
```
README.md → INTRODUCTION.md → GETTING_STARTED.md → TUTORIAL.md
```

**Intermediate Path**:
```
README.md → GETTING_STARTED.md → ALGORITHMS.md → API_REFERENCE.md
```

**Advanced Path**:
```
README.md → src/awb_system.py → API_REFERENCE.md → Custom implementation
```

## 💻 Code Files

### Core Implementation

**`src/awb_system.py`** (962 lines)
- Complete AWB system implementation
- All analyzers, fusion, temporal filtering
- Production-ready code
- Well-commented for learning

**Key Classes**:
- `AutoWhiteBalanceSystem` - Main orchestrator
- `HeuristicAnalyzer` - Gray world algorithm
- `FaceAwareAnalyzer` - Face-based AWB
- `SegmentationAnalyzer` - Region-based AWB
- `MLAnalyzer` - Machine learning AWB
- `HybridDecisionFusion` - Trust-based fusion
- `TemporalFilter` - Video smoothing
- `CCMSelector` - Color correction matrix

### Example Scripts

**`examples/basic_usage.py`**
- Simplest AWB usage example
- Perfect for beginners
- Shows complete workflow
- Well-commented

**`examples/run.sh`**
- Easy execution script
- Handles virtual environment
- Interactive menu
- Multiple options

### Test Scripts

**`tests/test_real_data.py`**
- Real RAW10 sensor data processing
- Complete pipeline demonstration
- Production validation
- Detailed output

## 🧪 Test Data

**`data/*.raw`**
- Real RAW10 sensor data files
- Multiple resolutions
- Different lighting conditions
- Production-representative

## 📊 Results

**`results/`**
- Output images and comparisons
- JSON metadata
- Performance metrics
- Visual validation

## 🔍 Code Organization Principles

### 1. Separation of Concerns

- **Core logic**: `src/` - Pure AWB algorithms
- **Examples**: `examples/` - Usage demonstrations
- **Tests**: `tests/` - Validation scripts
- **Data**: `data/` - Test inputs
- **Docs**: `docs/` - Documentation

### 2. Progressive Complexity

- **Simple**: `examples/basic_usage.py`
- **Intermediate**: `docs/TUTORIAL.md`
- **Advanced**: `src/awb_system.py`

### 3. Self-Documenting

- Clear file names
- Comprehensive comments
- Type hints
- Docstrings

### 4. Educational Focus

- Step-by-step tutorials
- Concept explanations
- Visual examples
- Real-world applications

## 🎓 Learning Resources

### By Topic

**White Balance Concepts**:
- `docs/INTRODUCTION.md`
- `docs/ALGORITHMS.md` (Gray World section)

**Implementation**:
- `docs/TUTORIAL.md`
- `src/awb_system.py` (with comments)

**Usage**:
- `docs/GETTING_STARTED.md`
- `examples/basic_usage.py`
- `docs/USAGE.md`

**Advanced Topics**:
- `docs/ALGORITHMS.md`
- `docs/API_REFERENCE.md`
- `src/awb_system.py` (full implementation)

## 🔧 Customization Points

### Easy to Modify

1. **Add Analyzer**: Create new class inheriting `AWBAnalyzer`
2. **Change Fusion**: Modify `HybridDecisionFusion`
3. **Adjust Filtering**: Tune `TemporalFilter` parameters
4. **Custom CCM**: Extend `CCMSelector`

### Integration Points

1. **Camera Pipeline**: Use `AutoWhiteBalanceSystem.process_frame()`
2. **Face Detection**: Pass `face_rois` parameter
3. **Segmentation**: Pass `segmentation_map` parameter
4. **Sensor Data**: Use `FrameStatsExtractor.extract_from_bayer_stats()`

## 📝 Code Style

- **Python 3.7+** compatible
- **Type hints** for clarity
- **Docstrings** for documentation
- **NumPy** for numerical operations
- **OpenCV** for image processing

## 🚀 Quick Navigation

### Find Code

- **Main system**: `src/awb_system.py`
- **Simple example**: `examples/basic_usage.py`
- **Real data test**: `tests/test_real_data.py`

### Find Documentation

- **Start here**: `README.md`
- **Concepts**: `docs/INTRODUCTION.md`
- **Tutorial**: `docs/TUTORIAL.md`
- **API**: `docs/API_REFERENCE.md`

### Find Examples

- **Basic**: `examples/basic_usage.py`
- **Real data**: `tests/test_real_data.py`
- **Usage patterns**: `docs/USAGE.md`

## ✅ Checklist for Book Readers

- [ ] Read `README.md` for overview
- [ ] Read `docs/INTRODUCTION.md` for concepts
- [ ] Run `examples/basic_usage.py` to see it work
- [ ] Read `docs/TUTORIAL.md` to understand implementation
- [ ] Study `src/awb_system.py` for production code
- [ ] Run `tests/test_real_data.py` for real validation
- [ ] Experiment with your own images
- [ ] Read `docs/ALGORITHMS.md` for deep understanding

---

**This organization makes it easy to:**
- Learn white balance concepts
- Understand the implementation
- Use the code in your projects
- Modify and extend the system

**Questions?** Check the documentation in `docs/` or read the code comments!

