# Project Kisan Data

The original backend expects a pre-built FAISS index and a directory of agricultural PDF resources.

The supplied implementation loads them from Kaggle paths. Those files are intentionally not included in this repository by default.

Set `LOCAL_FAISS_PATH` and `DIRECTORY_PATH` when running outside Kaggle if you maintain equivalent local resources.

The current application only loads the existing FAISS index; the original PDF ingestion/index-building cells were commented out.
