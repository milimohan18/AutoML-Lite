# Use the non-GUI "Agg" backend before anything imports matplotlib.pyplot. The default
# GUI backend (TkAgg) crashes when charts are drawn from background-job threads.
import matplotlib

matplotlib.use("Agg")
