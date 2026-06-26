"""
3DCNN_gui.py - Main GUI application for Deep3DCCS

This module implements the PyQt5-based graphical user interface for the Deep3DCCS system, 
which predicts Collision Cross-Section (CCS) values from molecular structures using a 
3D Convolutional Neural Network (3DCNN).

Key Components:
  - MyApp: The main application class inheriting from BaseClass (utility_modules.BaseClass)
  - Functions for molecular structure optimization from SMILES
  - Functions for generating 2D projections from 3D structures
  - Functions for training the 3DCNN model with configurable parameters
  - Functions for inference and evaluation of trained models

The GUI is organized into tabs for:
  1. Molecular Optimization: Convert SMILES to optimized 3D structures
  2. 2D Projection: Generate multi-view 2D projections from 3D structures
  3. Train 3DCNN: Configure and train the 3DCNN regression model
  4. Inference: Use trained models to predict CCS for new molecules

Usage:
  Run this script to launch the GUI. Ensure all dependencies are installed and 
  the required data directories (datasets, models, etc.) are set up.

Dependencies: PyQt5, TensorFlow, RDKit, OpenCV, NumPy, Pandas, etc.

Author: Siriraj Metabolomics & Phenomics Center (SiMPC), Mahidol University
"""

import os, sys
import time, glob
import re

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = (
    "1"  # Set the environment variable for deterministic TensorFlow operations
)
import csv
from csv import reader

# =========
import tensorflow as tf

tf.get_logger().setLevel("ERROR")  # or 'WARNING'
import numpy as np
import math
from PIL import Image
import seaborn as sns
import warnings
import random
import datetime
import pickle
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.losses import Huber
from tensorflow.keras.layers import (
    Conv3D,
    MaxPooling3D,
    Flatten,
    Dense,
    Dropout,
    Add,
    Input,
    BatchNormalization,
    LeakyReLU,
    GlobalAveragePooling3D,
)

from scipy.stats import pearsonr, linregress
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
import gc
from tqdm import tqdm
from colorama import *
from termcolor import colored
import json
import cv2
import pandas as pd

# ==============================
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtGui import QPixmap, QImage, QColor, QPainter
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QListWidget, QTableWidgetItem
import PyQt5.QtWidgets

# ============================= for 3D visualization
from mpl_toolkits.mplot3d import Axes3D
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw, AllChem
from rdkit.Chem import rdMolDescriptors
import math as m

# ==============================
cur_path = os.path.dirname(os.path.abspath(__file__))  # current absolute path
os.chdir(cur_path)  # change the working path to current working dir
print(colored("\n#Current working directory :%s" % cur_path, "blue"))  # show working directory
# ================== Import external modules from cur_path
from _core import *
from helper_tools import *

# ===================Resize Image (based on width fit or height fit) CANNOT USE FOR THE CV2. Resize for 256 x 256 as it can make non symmetric output


# change the datafiel header names in final filtered file to fit for the training
def rename_column_headers(target_file):
    # Load the DataFrame
    df = pd.read_csv(target_file, encoding="latin-1")

    # Rename columns: change 'CCS' to 'exp_ccs'
    df.rename(columns={"CCS": "exp_ccs"}, inplace=True)

    # Save the modified DataFrame (optional)
    df.to_csv(file_path, index=False)

    return df


# return time elapsed in seconds
def timed_elapsed_in_min(start_time):
    end_time = datetime.datetime.now()  # Record end time
    elapsed_time = (end_time - start_time).total_seconds() / 60  # Convert to minutes
    return elapsed_time


"""
func: filterd_source_file() | This script will create a new CSV file named "processed.csv" in the "./data" folder, containing only the rows from the 
source file where the corresponding "relative_percentage_error" in the results file is less than or equal to 3.0.
This solution addresses the prompt's requirements effectively by filtering the source data based on the error 
threshold in the results file and maintaining the original format of the source data.
"""


def filterd_source_file(source_file, result_file, output_file, output_trainer, error_threshold=3.0):
    """
    Filters source data based on matching 'ID' values and an optional error threshold in the result data.

    Args:
        source_file: Path to the source CSV file.
        result_file: Path to the result CSV file.
        output_file: Path to the output CSV file.
        error_threshold: Threshold for 'relative_percentage_error' in the result data.
    Returns:
        None ,  (len(source_df) - len(filtered_source_df) )
        # Example usage:
    filterd_source_file( source_file="./data/source.csv", result_file="./data/result.csv", output_file="./data/output.csv" )
    """

    if source_file.lower().endswith(".xls") or source_file.lower().endswith(
        ".xlsx"
    ):  # Load data from the Excel file
        print("Reading the data as miscrosoft excel file")
        source_df = pd.read_excel(source_file, sheet_name=0, engine="openpyxl")
    elif source_file.endswith(".csv"):
        print("Reading the data as csv file")
        source_df = pd.read_csv(
            source_file, encoding="latin-1"
        )  # 1. Loading source data from '{source_file}'")
    else:
        print("Error!. 'Cannot load source file with SMILES. Inefrence aborted")
        return

    result_df = pd.read_csv(
        result_file, encoding="latin-1"
    )  # 2. Loading result data from '{result_file}'")

    try:
        filtered_result_df = result_df[
            result_df["relative_percentage_error"] > error_threshold
        ]  # 3. Filtering result data based on 'relative_percentage_error'")
    except KeyError:
        print(
            "Error! 'relative_percentage_error' column not found in result.csv. Filtering skipped."
        )
        filtered_result_df = result_df
        return None

    # print("debug: ", filtered_result_df.head())
    result_values = [
        item.upper() for item in set(filtered_result_df["ID"])
    ]  # 4. Extracting unique 'ID' values from filtered result data")

    filtered_source_df = source_df[
        ~source_df["ID"].isin(result_values)
    ]  # Filtering source data: Removing rows with 'ID' in result_values")

    print(f"Saving filtered source data to '{output_file}'")
    filtered_source_df.to_csv(output_file, index=False)

    # for direct Training, Rename columns: change 'CCS' to 'exp_ccs' and "m/z" to "mz_ratio"
    filtered_source_df.rename(columns={"exp_ccs": "exp_ccs", "mz_ratio": "mz_ratio"}, inplace=True)

    print(f"Saving filtered for training source data to '{output_file}'")
    filtered_source_df.to_csv(output_trainer, index=False)

    return len(source_df) - len(filtered_source_df)  # returns the numbero samples removed,


# Load pyQt5 modules for GUIs & more
from utility_modules import BaseClass


class MyApp(BaseClass):
    def __init__(self):
        super(MyApp, self).__init__()
        # self.setupUi(self)
        self.post_train_evalulation = None
        self.reg_model_fname = ""
        # =========================================
        show_logo()
        # =========================================
        self.qm = QtGui.QMessageBox
        # =========================================
        self.gfx1 = self.plot_val_mae.addPlot()  # for mean absoluete error
        self.gfx2 = self.plot_val_mape.addPlot()  # for mean absolute percetage error
        # =========================================
        self.initalize_vars()
        # Force Train/Test/Validation sliders to allow 0-100% for data saturation analysis.
        self.configure_split_sliders()
        self.update_gui_vars()
        # Re-apply after update_gui_vars() in case GUI/config logic changes slider limits.
        self.configure_split_sliders()
        self.set_inf_table_header()  # set header table
        # self.stay_on_top(stay_top_flag =True)  # Stay on top deactivated for now

    def configure_split_sliders(self):
        """
        Force Train/Test/Validation sliders to allow 0-100%.

        This overrides slider limits stored in the .ui file. It is needed for
        data saturation analysis, where the Train slider may need values below
        its original GUI minimum.
        """
        split_sliders = [self.train_ratio_slider, self.test_ratio_slider, self.val_ratio_slider]

        for slider in split_sliders:
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setSingleStep(1)
            slider.setPageStep(5)
            slider.setTickInterval(10)

    def set_inf_table_header(self):
        self.tableWidget.clear()  # reset the table widgets
        self.tableWidget.setRowCount(0)  # Remove all rows
        self.tableWidget.setColumnCount(0)  # Remove all columns
        self.inf_table_Header = ["molecule", "SMILE", "id/label", "mz ratio", "pred.CCS", "exp.CCS"]
        self.tableWidget.setStyleSheet("QTableWidget { background-color: #FFFFE0; }")
        self.tableWidget.setColumnCount(len(self.inf_table_Header))
        self.tableWidget.setHorizontalHeaderLabels(self.inf_table_Header)

    def clear_gpu_mem(self):
        try:
            del self.regression_model
        except:
            pass
        tf.keras.backend.clear_session()
        gc.collect()
        print(colored(f"$Memory cleared before processing...", "green"))

    def stay_on_top(self, stay_top_flag=True):
        if stay_top_flag:
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
            self.setAttribute(Qt.WA_AlwaysStackOnTop, stay_top_flag)
        else:
            self.setAttribute(Qt.WA_AlwaysStackOnTop, stay_top_flag)
        self.show()  # Re-show the window to apply the new on-top setting

    def add_rows_to_table(self, rows):
        self.set_inf_table_header()
        for row_data in rows:
            row_position = self.tableWidget.rowCount()
            self.tableWidget.insertRow(row_position)
            for column, data in enumerate(row_data):
                self.tableWidget.setItem(row_position, column, QTableWidgetItem(data))

    # To store the shown window as an image file
    def save_win_image(self, fname="train_losses.png"):
        pixmap = QPixmap(self.win.size())
        painter = QPainter(pixmap)  # Begin painting on the pixmap
        self.win.render(painter)  # Render the content of the window onto the pixmap
        painter.end()  # End the painting
        # Save the pixmap as an image file
        pixmap.save(fname)

    def console_show_store_vars(self):
        print("\n" + "=" * 80)
        print(
            colored("Runtime Variables, Configurations & Hyper parameters", "cyan", attrs=["bold"])
        )
        print("=" * 80)

        variables = {
            # ====================== DATA & PATHS ======================
            "system verbosity": self.sys_verbosity.currentText(),
            "Allow gpu memory growth": self.set_gpu_growth.isChecked(),
            "csv_file_path": self.csv_file_path,
            "dataset_path": self.dataset_path,
            "SMILE_src_filepath": self.SMILE_src_filepath,
            "sdf_mol_dirpath": self.sdf_mol_dirpath,
            "projection_output": self.projection_output,
            "optimized_struct_output": self.optimized_struct_output_dirpath,
            "inf_filepath": self.inf_filepath,
            "inference_projection_path": self.inference_projection_path.toPlainText(),
            "eval_dirpath": self.eval_dirpath,
            "evaluations_dir": self.evaluations_dir,
            "config_file_path": self.config_file_path,
            # ====================== MODEL FILES ======================
            "Base_3dccs_model": self.Base_3dccs_model,
            "basemodel_flag": self.basemodel_flag,
            "base_reg_model": self.base_reg_model,
            "model_config": self.model_config,
            "store_weights": self.store_weights,
            "show model summary": self.model_summary_flag,
            "dataset_id for Train|Test": self.dataset_id,
            "eval dir for Train|Test": self.eval_dirname,
            # ====================== TRAINING ======================
            "train_epoch": self.train_epoch,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "lr_decay_epoch": self.lr_decay_epoch,
            "lr_decay_frac": self.lr_decay_frac,
            "use_lr_decay": self.use_lr_decay,
            "random_seed": self.random_seed,
            "sample_limit": self.sample_limit,
            "num_rotation": self.num_rotation,
            "train_singlemode": self.train_singlemode,
            # ====================== MODEL PARAMS ======================
            "activation_func": self.activation_func,
            "loss_func": self.loss_func,
            "relu_alpha": self.relu_alpha,
            "set_precision": self.set_precision,
            # ====================== IMAGE / DATA PROCESSING ======================
            "img_dim": self.img_dim,
            "image_threshold": self.image_threshold,
            "use_thresthold_gui": self.use_thresthold_gui,
            "use_multipixel_flag": self.use_multipixel_flag,
            "min_cutoff_weight": self.min_cutoff_weight,
            "max_cutoff_weight": self.max_cutoff_weight,
            # ====================== FLAGS ======================
            "post_train_evalulation": self.post_train_evalulation,
            "use_computed_mass": self.use_computed_mass,
            "AllCCSID_name_flag": self.AllCCSID_name_flag,
            "molecule_name_flag": self.molecule_name_flag,
            "inf_adduct_type": self.inf_adduct_type,
            "autoset missing exp. ccs": self.autoset_expccs_flag,
            "sort raw SMILE input data": self.sort_dataset_flag,
            # ====================== GPU ======================
            "gpu_index": self.gpu_index,
            "gpu_name": self.gpu_name,
            "set_gpu_growth": self.set_gpu_growth.isChecked(),
            # ====================== IDENTIFIERS ======================
            "timestamp": self.timestamp,
            "exp_counter": self.exp_counter,
            "run_mode": self.run_mode,
        }

        # Pretty print
        for key, value in variables.items():
            print(colored(f"{key:<30}", "yellow") + f": {value}")

        print("=" * 80 + "\n")

    def _timing_safe_value(self, value):
        """Return a simple value that is safe to write into Excel."""
        try:
            if isinstance(value, (np.integer, np.floating)):
                return value.item()
        except Exception:
            pass
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        return value

    def _timing_output_dir(self, preferred_dir=None):
        """Choose a valid output directory for timing reports."""
        candidates = [preferred_dir, getattr(self, "evaluations_dir", None), cur_path]
        for cand in candidates:
            if cand is None:
                continue
            cand = str(cand).strip()
            if cand == "":
                continue
            try:
                os.makedirs(cand, exist_ok=True)
                return cand
            except Exception:
                continue
        os.makedirs(cur_path, exist_ok=True)
        return cur_path

    def write_runtime_timing_excel(
        self, stage_name, summary_rows=None, detail_rows=None, output_dir=None, filename_prefix=None
    ):
        """
        Write one separate Excel file for a GUI-clicked runtime stage.

        stage_name examples:
          - optimized_structure
          - 2d_projection
          - training
          - inference
        """
        output_dir = self._timing_output_dir(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        timestamp = getattr(self, "timestamp", None)
        if timestamp in [None, ""]:
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

        safe_stage = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(stage_name)).strip("_")
        if filename_prefix is None or str(filename_prefix).strip() == "":
            filename_prefix = getattr(self, "dataset_id", "Deep3DCCS")
        safe_prefix = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(filename_prefix)).strip("_")

        excel_path = os.path.join(
            output_dir, f"{safe_prefix}_{safe_stage}_runtime_timing_{timestamp}.xlsx"
        )

        summary_rows = summary_rows or []
        detail_rows = detail_rows or []

        summary_df = pd.DataFrame(summary_rows)
        detail_df = pd.DataFrame(detail_rows)

        # Convert numpy values to plain Python values for clean Excel output.
        if not summary_df.empty:
            summary_df = summary_df.applymap(self._timing_safe_value)
        if not detail_df.empty:
            detail_df = detail_df.applymap(self._timing_safe_value)

        try:
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                if summary_df.empty:
                    summary_df = pd.DataFrame([{"message": "No summary rows were recorded."}])
                summary_df.to_excel(writer, sheet_name="summary", index=False)

                if detail_df.empty:
                    detail_df = pd.DataFrame([{"message": "No detail rows were recorded."}])
                detail_df.to_excel(writer, sheet_name="details", index=False)
            print(colored(f"Runtime timing Excel stored at: {excel_path}", "green"))
        except Exception as e:
            print(colored(f"WARNING! Could not save runtime timing Excel file: {e}", "red"))

        return excel_path

    def compute_exact_mass(self, smiles):
        # Parse the SMILES string to a molecule
        molecule = Chem.MolFromSmiles(smiles)

        if molecule is None:
            print(colored("\n#ERROR !Invalid SMILES string provided", "red"))
            return -1
        else:
            # Calculate the exact molecular mass
            exact_mass = Descriptors.ExactMolWt(molecule)

        return exact_mass

    # This function is just for rough preview of moleculer structrue, nothing to do with orignal structure code provided by the team
    def show_3d_molecules_preview(self, sdf_file_path, num_to_display=1):  # show 3D moleculae

        def plot_3d_molecule(molecule):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            mol_name = os.path.basename(sdf_file_path)
            mol_name = os.path.splitext(mol_name)[0]  # get only molecule name name
            fig.canvas.manager.set_window_title(f"3D view: {mol_name}")
            conformer = molecule.GetConformer()
            atoms = molecule.GetAtoms()
            # Extract coordinates of atoms
            xs = [conformer.GetAtomPosition(atom.GetIdx()).x for atom in atoms]
            ys = [conformer.GetAtomPosition(atom.GetIdx()).y for atom in atoms]
            zs = [conformer.GetAtomPosition(atom.GetIdx()).z for atom in atoms]
            elements = [atom.GetSymbol() for atom in atoms]

            # Plot the atoms
            ax.scatter(xs, ys, zs, c="r", s=100)  # Change 'r' to a different color if needed

            # Annotate the atoms with their element symbols
            for x, y, z, element in zip(xs, ys, zs, elements):
                ax.text(x, y, z, element, fontsize=12)

            # Plot bonds
            for bond in molecule.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                x = [xs[begin_idx], xs[end_idx]]
                y = [ys[begin_idx], ys[end_idx]]
                z = [zs[begin_idx], zs[end_idx]]
                ax.plot(x, y, z, color="b")  # Change 'b' to a different color if needed
            ax.set_xlabel("X-axis")
            ax.set_ylabel("Y-axis")
            ax.set_zlabel("Z-axis")
            plt.show()
            # ================================

        # Read molecules from the SDF file
        supplier = Chem.SDMolSupplier(sdf_file_path)
        molecules = [mol for mol in supplier if mol is not None]

        # == multiple Structure view check
        if self.allow_multiple_preview_flag.isChecked():
            pass
        else:
            plt.close("all")

        if not molecules:
            print("No molecules found in the SDF file.")
            return

        # Generate 3D coordinates for the molecules
        for mol in molecules:
            if mol is not None:
                AllChem.EmbedMolecule(mol)

        # Plot the first few molecules in 3D using Matplotlib
        for mol in molecules[:num_to_display]:
            if mol is not None:
                plot_3d_molecule(mol)

    # this function is just for rough preview of moleculer structrue, nothing to do with orignal structure code provided by the team
    def show_preview_molecules(self, sdf_file_path, num_to_display=5):
        # Read molecules from the SDF file
        supplier = Chem.SDMolSupplier(sdf_file_path)
        molecules = [mol for mol in supplier if mol is not None]

        if not molecules:
            print(
                colored(
                    "Warning! invalid SDF datafile or No molecules found in the SDF file.", "red"
                )
            )
            return

        # Set up RDKit drawing options
        draw_options = Draw.MolDrawOptions()
        draw_options.bgColor = (0, 0, 0)  # Set background to black
        draw_options.color = (1, 1, 1)  # Set text color to white

        # Create a drawer with custom options
        drawer = Draw.MolDraw2DCairo(200 * 3, 200 * (num_to_display // 3 + 1))
        drawer.SetDrawOptions(draw_options)
        drawer.DrawMolecules(molecules[:num_to_display])
        drawer.FinishDrawing()

        # Convert the drawing to a NumPy array
        img = drawer.GetDrawingText()
        img_array = np.frombuffer(img, np.uint8)
        img_cv = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        img_cv = image_resize(img_cv, height=211)  # resize to fit
        self.mol_formula_preview.setPixmap(QPixmap.fromImage(self.displayImage(img_cv)))
        self.mol_formula_preview.setAlignment(QtCore.Qt.AlignCenter)

        (
            self.show_3d_molecules_preview(sdf_file_path)
            if self.enable_3d_preview_flag.isChecked()
            else None
        )

        return

    #  This part is mainly adopted (imported) from original code to process 3D structure that you might need to focus

    def build_otptimized_structure_from_smile(
        self, input_smile_datafile=None
    ):  # Build optimized molecule structure from SMiles

        stage_start_perf = time.perf_counter()
        timing_records = []

        file_type = os.path.splitext(input_smile_datafile)[1]

        if file_type == ".xls" or file_type == ".xlsx":
            df = pd.read_excel(input_smile_datafile, sheet_name=0, engine="openpyxl")
        elif file_type == ".csv":
            df = pd.read_csv(input_smile_datafile, encoding="latin-1")
        else:
            print("Error reading SMILEs data ! Only .csv and .xls (.xlsx) are currently supported")
            return

        if self.sort_dataset_flag:
            print(
                colored(
                    "Dataset rows are being alphanumerically sorted based on adduct-labels. Same as MS Excel’s Custom Sort (A–Z) behavior sort",
                    "yellow",
                )
            )
            df = df.sort_values(by="ID", ascending=True, kind="mergesort")
            print("\nSorted dataset sample(s) preview:")
            print("\n--------------------------------------------")
            print(colored(df.head(), "blue"))
            print("\n--------------------------------------------")

        data_tag = os.path.splitext(os.path.basename(input_smile_datafile))[0]
        self.struct_output_dir = os.path.join(
            self.optimized_struct_output_dirpath, data_tag + "_optimized_structure"
        )
        os.makedirs(self.struct_output_dir, exist_ok=True)

        self.list_optimized_mols.clear()
        count = 0
        index = 0
        total_samples = len(df)
        skipped_mol = 0

        for index, row in tqdm(df.iterrows(), desc="Processing SMILEs data", ncols=200):
            molecule_start_perf = time.perf_counter()
            molecule_name = row["name"] if self.AllCCSID_name_flag == False else row["ID"]
            smiles = row["SMILES"]
            status = "success"
            error_message = ""
            sdf_output_file = ""
            xyz_output_file = ""

            try:
                mol = Chem.MolFromSmiles(smiles)
            except Exception as e:
                mol = None
                error_message = str(e)

            if mol is None:
                print(colored(f"\nInvalid SMILES for molecule {molecule_name}. Skipping...", "red"))
                skipped_mol += 1
                status = "failed_invalid_smiles"
                timing_records.append(
                    {
                        "stage": "convert_smiles_to_optimized_structure",
                        "source_file": input_smile_datafile,
                        "index": index,
                        "ID_or_name": molecule_name,
                        "SMILES": smiles,
                        "status": status,
                        "error_message": error_message,
                        "elapsed_min": (time.perf_counter() - molecule_start_perf) / 60.0,
                        "output_sdf": sdf_output_file,
                        "output_xyz": xyz_output_file,
                    }
                )
                continue

            try:
                mol = Chem.AddHs(mol)
                params = AllChem.ETKDGv3()
                params.randomSeed = self.random_seed
                params.maxAttempts = 500
                params.useRandomCoords = True

                embed_result = AllChem.EmbedMolecule(mol, params)
                AllChem.UFFOptimizeMolecule(mol, maxIters=2000)

                sdf_output_file = os.path.join(
                    self.struct_output_dir, f"{molecule_name}_optimized_structure.sdf"
                )
                w = Chem.SDWriter(sdf_output_file)
                w.write(mol)
                w.close()
                self.list_optimized_mols.addItem(str(molecule_name))

                xyz_output_file = os.path.join(self.struct_output_dir, f"{molecule_name}.txt")
                with open(xyz_output_file, "w", encoding="utf-8") as f:
                    for atom in mol.GetAtoms():
                        pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
                        f.write(f"{atom.GetSymbol()} {pos.x:.4f} {pos.y:.4f} {pos.z:.4f}\n")

                count += 1
                self.update_progressBar(total_samples, index + 1)

            except Exception as e:
                skipped_mol += 1
                status = "failed_optimization_or_write"
                error_message = str(e)

            timing_records.append(
                {
                    "stage": "convert_smiles_to_optimized_structure",
                    "source_file": input_smile_datafile,
                    "index": index,
                    "ID_or_name": molecule_name,
                    "SMILES": smiles,
                    "status": status,
                    "error_message": error_message,
                    "embed_result": locals().get("embed_result", ""),
                    "elapsed_min": (time.perf_counter() - molecule_start_perf) / 60.0,
                    "output_sdf": sdf_output_file,
                    "output_xyz": xyz_output_file,
                }
            )

        total_elapsed_min = (time.perf_counter() - stage_start_perf) / 60.0

        print(
            colored(
                f"Optimized structures count   :  {count}/{index+1} | {round(100*count/(index+1),2)}%",
                "yellow",
            )
        )
        print(
            colored(
                f"skipped molecule count       :  {skipped_mol}/{index+1} | {round(100*skipped_mol/(index+1),2)}%",
                "red",
            )
        )
        if count > 0:
            print(colored(f"Processed Structures saved   :  {self.struct_output_dir}", "yellow"))

        successful_times = [
            r["elapsed_min"] for r in timing_records if r.get("status") == "success"
        ]
        summary_rows = [
            {
                "stage": "convert_smiles_to_optimized_structure",
                "source_file": input_smile_datafile,
                "output_dir": self.struct_output_dir,
                "total_input_rows": total_samples,
                "successful_count": count,
                "skipped_count": skipped_mol,
                "total_elapsed_min": total_elapsed_min,
                "mean_elapsed_min_per_successful_molecule": (
                    float(np.mean(successful_times)) if successful_times else np.nan
                ),
                "sd_elapsed_min_per_successful_molecule": (
                    float(np.std(successful_times, ddof=1)) if len(successful_times) > 1 else np.nan
                ),
                "random_seed": self.random_seed,
            }
        ]
        self.write_runtime_timing_excel(
            stage_name="optimized_structure",
            summary_rows=summary_rows,
            detail_rows=timing_records,
            output_dir=self.struct_output_dir,
            filename_prefix=data_tag,
        )

        if self.set_autoset_2D_projection_flag.isChecked():
            self.sdf_mol_input_dirpath.setText(self.struct_output_dir)
            os.makedirs(self.struct_output_dir, exist_ok=True)
            self.projection_output_dirpath.setText(
                os.path.join(self.struct_output_dir, "2D_projections")
            )
            os.makedirs(os.path.join(self.struct_output_dir, "2D_projections"), exist_ok=True)

    def build_2D_projections(self, sdf_mol_dirpath=None):
        import matplotlib

        original_plt_backend = matplotlib.get_backend()
        matplotlib.use("Agg")

        stage_start_perf = time.perf_counter()
        timing_records = []

        def Rx(theta):
            return np.matrix(
                [[1, 0, 0], [0, m.cos(theta), -m.sin(theta)], [0, m.sin(theta), m.cos(theta)]]
            )

        def Ry(theta):
            return np.matrix(
                [[m.cos(theta), 0, m.sin(theta)], [0, 1, 0], [-m.sin(theta), 0, m.cos(theta)]]
            )

        def Rz(theta):
            return np.matrix(
                [[m.cos(theta), -m.sin(theta), 0], [m.sin(theta), m.cos(theta), 0], [0, 0, 1]]
            )

        max_x, min_x = 15, -15
        max_y, min_y = 15, -15
        num_rotation = self.num_rotation
        theta = m.pi / num_rotation

        AtomInfo = {
            "Element": [
                "H",
                "He",
                "Li",
                "Be",
                "B",
                "C",
                "N",
                "O",
                "F",
                "Ne",
                "Na",
                "Mg",
                "Al",
                "Si",
                "P",
                "S",
                "Cl",
                "Ar",
                "K",
                "Ca",
                "Ga",
                "Ge",
                "As",
                "Se",
                "Br",
                "Kr",
                "Rb",
                "Sr",
                "In",
                "Sn",
                "Sb",
                "Te",
                "I",
                "Xe",
                "Cs",
                "Ba",
                "Tl",
                "Pb",
                "Bi",
                "Po",
                "At",
                "Rn",
                "Fr",
                "Ra",
            ],
            "VdW Radius": [
                1.10,
                1.40,
                1.81,
                1.53,
                1.92,
                1.70,
                1.55,
                1.52,
                1.47,
                1.54,
                2.27,
                1.73,
                1.84,
                2.10,
                1.80,
                1.80,
                1.75,
                1.88,
                2.75,
                2.31,
                1.87,
                2.11,
                1.85,
                1.90,
                1.83,
                2.02,
                3.03,
                2.49,
                1.93,
                2.17,
                2.06,
                2.06,
                1.98,
                2.16,
                3.43,
                2.68,
                1.96,
                2.02,
                2.07,
                1.97,
                2.02,
                2.20,
                3.48,
                2.83,
            ],
            "MW": [
                1.00797,
                4.0026,
                6.941,
                9.01218,
                10.81,
                12.011,
                14.0067,
                15.9994,
                18.998403,
                20.179,
                22.98977,
                24.305,
                26.98154,
                28.0855,
                30.97376,
                32.06,
                35.453,
                39.948,
                39.0983,
                40.08,
                69.72,
                72.59,
                74.9216,
                78.96,
                79.904,
                83.8,
                85.4678,
                87.62,
                114.82,
                118.69,
                121.75,
                127.6,
                126.9045,
                131.3,
                132.9054,
                137.33,
                204.37,
                207.2,
                208.9804,
                209,
                210,
                222,
                223,
                226.0254,
            ],
        }
        Atom = AtomInfo["Element"]
        VdWRadius = AtomInfo["VdW Radius"]
        MW = AtomInfo["MW"]

        count = 0
        self.projection_2d_list.clear()
        optimized_samples = glob.glob(os.path.join(sdf_mol_dirpath, "*.txt"))
        total_samples = len(optimized_samples)
        print(colored(f"Total optimized structures found : {total_samples}", "yellow"))

        try:
            for file_index, file in enumerate(
                tqdm(optimized_samples, desc="Building 2D projection", ncols=100)
            ):
                sample_start_perf = time.perf_counter()
                motherfol = os.path.basename(os.path.splitext(file)[0])
                status = "success"
                error_message = ""
                x_time = y_time = z_time = np.nan

                try:
                    self.projection_2d_list.addItem(os.path.basename(file))

                    with open(file, "r") as infile:
                        lines = infile.readlines()
                        count += 1

                    df = pd.DataFrame(list(reader(lines)))
                    df.dropna(inplace=True)
                    df = df[0].str.split(r"\s+", expand=True)
                    df.columns = ["atom", "x", "y", "z"]

                    x = [float(value) for value in df["x"].values.tolist()]
                    y = [float(value) for value in df["y"].values.tolist()]
                    z = [float(value) for value in df["z"].values.tolist()]
                    element = df["atom"]

                    collect_mx, collect_my, collect_mz, collect_MW = 0, 0, 0, 0
                    radius = []

                    for i in range(0, len(element)):
                        atom_index = Atom.index(element[i])
                        radius.append(VdWRadius[atom_index])
                        dummy_MW = MW[atom_index]
                        collect_mx += dummy_MW * x[i]
                        collect_my += dummy_MW * y[i]
                        collect_mz += dummy_MW * z[i]
                        collect_MW += dummy_MW

                    CM_x = collect_mx / collect_MW
                    CM_y = collect_my / collect_MW
                    CM_z = collect_mz / collect_MW

                    x = [i - CM_x for i in x]
                    y = [i - CM_y for i in y]
                    z = [i - CM_z for i in z]

                    points_whole_ax = 5 * 0.8 * 72
                    radius = np.array(radius)
                    point_radius = []
                    axis_length = max_x - min_x
                    for i in range(0, len(radius)):
                        dummy = 2 * radius[i] / axis_length * points_whole_ax
                        dummy = dummy**2
                        point_radius.append(dummy)

                    os.makedirs(os.path.join(self.projection_output, motherfol), exist_ok=True)
                    subfolx = os.path.join(self.projection_output, motherfol, "x")
                    subfoly = os.path.join(self.projection_output, motherfol, "y")
                    subfolz = os.path.join(self.projection_output, motherfol, "z")
                    os.makedirs(subfolx, exist_ok=True)
                    os.makedirs(subfoly, exist_ok=True)
                    os.makedirs(subfolz, exist_ok=True)

                    axis_start = time.perf_counter()
                    rotateangle = -theta
                    for j in range(0, num_rotation):
                        rotateangle = rotateangle + theta
                        coor = np.array([x, y, z]).transpose()
                        test = coor * Rx(rotateangle)
                        x_new, y_new, z_new = test.T
                        x_new = x_new.tolist()
                        y_new = y_new.tolist()
                        plt.figure(figsize=[5, 5])
                        ax = plt.axes(
                            [0.1, 0.1, 0.8, 0.8], xlim=(min_x, max_x), ylim=(min_y, max_y)
                        )
                        ax.scatter(x_new, y_new, s=point_radius, color="black")
                        plt.axis("off")
                        writename = os.path.join(subfolx, motherfol + "_x_" + str(j) + ".png")
                        plt.savefig(writename)
                        plt.close()
                    x_time = (time.perf_counter() - axis_start) / 60.0

                    axis_start = time.perf_counter()
                    rotateangle = -theta
                    for j in range(0, num_rotation):
                        rotateangle = rotateangle + theta
                        coor = np.array([x, y, z]).transpose()
                        test = coor * Ry(rotateangle)
                        x_new, y_new, z_new = test.T
                        x_new = x_new.tolist()
                        y_new = y_new.tolist()
                        z_new = z_new.tolist()
                        plt.figure(figsize=[5, 5])
                        ax = plt.axes(
                            [0.1, 0.1, 0.8, 0.8], xlim=(min_x, max_x), ylim=(min_y, max_y)
                        )
                        ax.scatter(x_new, y_new, s=point_radius, color="black")
                        plt.axis("off")
                        writename = os.path.join(subfoly, motherfol + "_y_" + str(j) + ".png")
                        plt.savefig(writename)
                        plt.close()
                    y_time = (time.perf_counter() - axis_start) / 60.0

                    axis_start = time.perf_counter()
                    rotateangle = -theta
                    for j in range(0, num_rotation):
                        rotateangle = rotateangle + theta
                        coor = np.array([x, y, z]).transpose()
                        test = coor * Rz(rotateangle)
                        x_new, y_new, z_new = test.T
                        z_new = z_new.tolist()
                        y_new = y_new.tolist()
                        plt.figure(figsize=[5, 5])
                        ax = plt.axes(
                            [0.1, 0.1, 0.8, 0.8], xlim=(min_x, max_x), ylim=(min_y, max_y)
                        )
                        ax.scatter(z_new, y_new, s=point_radius, color="black")
                        plt.axis("off")
                        writename = os.path.join(subfolz, motherfol + "_z_" + str(j) + ".png")
                        plt.savefig(writename)
                        plt.close()
                    z_time = (time.perf_counter() - axis_start) / 60.0

                    self.update_progressBar(total_samples, file_index + 1)

                except Exception as e:
                    status = "failed"
                    error_message = str(e)

                timing_records.append(
                    {
                        "stage": "make_2D_projection_images",
                        "input_txt_file": file,
                        "sample_folder": motherfol,
                        "status": status,
                        "error_message": error_message,
                        "num_rotation_per_axis": num_rotation,
                        "total_projection_images_expected": 3 * num_rotation,
                        "x_axis_elapsed_min": x_time,
                        "y_axis_elapsed_min": y_time,
                        "z_axis_elapsed_min": z_time,
                        "total_elapsed_min": (time.perf_counter() - sample_start_perf) / 60.0,
                        "projection_output_dir": os.path.join(self.projection_output, motherfol),
                    }
                )

            print(
                colored(
                    f"2D projection build count :  {count}/{file_index+1} | {round(100*count/(file_index+1),2)}%",
                    "yellow",
                )
            )
            print(colored(f"2D projections saved      :  {self.projection_output}", "yellow"))

        finally:
            matplotlib.use(original_plt_backend)

        total_elapsed_min = (time.perf_counter() - stage_start_perf) / 60.0
        successful_times = [
            r["total_elapsed_min"] for r in timing_records if r.get("status") == "success"
        ]
        summary_rows = [
            {
                "stage": "make_2D_projection_images",
                "input_optimized_structure_dir": sdf_mol_dirpath,
                "projection_output_dir": self.projection_output,
                "total_txt_files_found": total_samples,
                "successful_count": sum(1 for r in timing_records if r.get("status") == "success"),
                "failed_count": sum(1 for r in timing_records if r.get("status") != "success"),
                "num_rotation_per_axis": num_rotation,
                "total_elapsed_min": total_elapsed_min,
                "mean_elapsed_min_per_successful_molecule": (
                    float(np.mean(successful_times)) if successful_times else np.nan
                ),
                "sd_elapsed_min_per_successful_molecule": (
                    float(np.std(successful_times, ddof=1)) if len(successful_times) > 1 else np.nan
                ),
            }
        ]
        prefix = (
            os.path.basename(os.path.normpath(sdf_mol_dirpath)) if sdf_mol_dirpath else "Deep3DCCS"
        )
        self.write_runtime_timing_excel(
            stage_name="2d_projection",
            summary_rows=summary_rows,
            detail_rows=timing_records,
            output_dir=self.projection_output,
            filename_prefix=prefix,
        )

        if self.autoset_3dccs_trainpath_flag.isChecked():
            self.train_msdata.setText(self.raw_smile_datafile.toPlainText())
            self.train_2d_projection_dirpath.setText(self.projection_output_dirpath.toPlainText())

    def compute_mz_ratio(
        self, smiles="", adduct=""
    ):  # computes the mz_ratio if not supplied. Adduct must be supplied to get correct prediction
        """Compute m/z from SMILES and adduct string.
        Supported adducts: [M+H]+  |  [M-H]- | [M+Na]+ | [M+HCOO]-
        """

        # Monoisotopic masses
        ADDUCT_TABLE = {"H": 1.007276, "Na": 22.989218, "HCOO": 44.998201}

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string.")

        M = rdMolDescriptors.CalcExactMolWt(mol)

        # --- Parse adduct string ---
        if not (adduct.startswith("[M") and (adduct.endswith("]+") or adduct.endswith("]-"))):
            raise ValueError("Unsupported adduct format.")

        # Determine charge sign
        charge = +1 if adduct.endswith("]+") else -1

        # Extract inside part: e.g. +H, -H, +Na, +HCOO
        inner = adduct[2:-2]  # remove "[M" and "]+" or "]-"

        if inner[0] not in ["+", "-"]:
            raise ValueError("Invalid adduct format.")

        sign = inner[0]
        group = inner[1:]

        if group not in ADDUCT_TABLE:
            print(f"Unsupported adduct group: {group}")

        mass_shift = ADDUCT_TABLE[group]

        # Apply correct sign to mass shift
        if sign == "+":
            mz = (M + mass_shift) / abs(charge)
        else:
            mz = (M - mass_shift) / abs(charge)

        return mz

    # this function is just for rough preview of moleculer structrue, nothing to do with orignal structure code provided by the team
    def show_2d_projections_preview(self, projection_dirpath=None):
        def load_images_from_folder(folder):
            images = []
            for filename in os.listdir(folder):
                if filename.endswith(".png"):
                    img = cv2.imread(os.path.join(folder, filename))
                    if img is not None:
                        images.append(img)
            return images

        def resize_images_to_common_height(images, common_height):
            resized_images = []
            for img in images:
                img = convert_rgb(
                    img, bg_color=(180, 180, 180)
                )  # convert backgroudn to RGB (180,180,180)
                aspect_ratio = img.shape[1] / img.shape[0]
                new_width = int(common_height * aspect_ratio)
                resized_img = cv2.resize(img, (new_width, common_height))
                resized_images.append(resized_img)
            return resized_images

        def concatenate_images_horizontally(images):
            return cv2.hconcat(images)

        def display_cyclic_motion_video(
            images, window_name="Cyclic Motion | ESC to Exit", delay=300
        ):
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            # Define the text and its properties
            text = os.path.splitext(self.projection_2d_list.currentItem().text())[
                0
            ]  # get only filename
            position = (10, 20)  # Position to put the text (x, y)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (0, 125, 0)  # Black color in BGR
            thickness = 2

            # Add the text to the image

            while True:
                for img in images:
                    cv2.putText(img, text, position, font, font_scale, color, thickness)
                    cv2.imshow(window_name, img)
                    if cv2.waitKey(delay) & 0xFF == 27:  # Press 'Esc' to exit
                        cv2.destroyAllWindows()
                        return

        for sample_axis in ["x", "y", "z"]:  #   loop through each axis  for preview

            images = load_images_from_folder(
                os.path.join(projection_dirpath, sample_axis)
            )  # load sample images

            if not images:
                print(colored("Error! No images exists in target folder.Skipping preview", "red"))
                return

            # Resize images to the height of the smallest image
            common_height = min(img.shape[0] for img in images)
            resized_images = resize_images_to_common_height(images, common_height)

            concatenated_image = concatenate_images_horizontally(resized_images)
            # cv2.imwrite(output_path, concatenated_image)

            (
                self.x_axis_projection_preview.setPixmap(
                    QPixmap.fromImage(self.displayImage(concatenated_image))
                )
                if sample_axis == "x"
                else None
            )
            (
                self.x_axis_projection_preview.setAlignment(QtCore.Qt.AlignCenter)
                if sample_axis == "x"
                else None
            )

            (
                self.y_axis_projection_preview.setPixmap(
                    QPixmap.fromImage(self.displayImage(concatenated_image))
                )
                if sample_axis == "y"
                else None
            )
            (
                self.y_axis_projection_preview.setAlignment(QtCore.Qt.AlignCenter)
                if sample_axis == "y"
                else None
            )

            (
                self.z_axis_projection_preview.setPixmap(
                    QPixmap.fromImage(self.displayImage(concatenated_image))
                )
                if sample_axis == "z"
                else None
            )
            (
                self.z_axis_projection_preview.setAlignment(QtCore.Qt.AlignCenter)
                if sample_axis == "z"
                else None
            )

            # Display the cyclic motion video
            try:
                plt.close("all")
                (
                    display_cyclic_motion_video(
                        resized_images,
                        window_name=f"Rotation preview @ {sample_axis}-axis | press ESC to exit",
                    )
                    if (self.x_3d_preview_flag.isChecked() and sample_axis == "x")
                    else None
                )
                (
                    display_cyclic_motion_video(
                        resized_images,
                        window_name=f"Rotation preview @ {sample_axis}-axis | press ESC to exit",
                    )
                    if (self.y_3d_preview_flag.isChecked() and sample_axis == "y")
                    else None
                )
                (
                    display_cyclic_motion_video(
                        resized_images,
                        window_name=f"Rotation preview @ {sample_axis}-axis | press ESC to exit",
                    )
                    if (self.y_3d_preview_flag.isChecked() and sample_axis == "z")
                    else None
                )
            except:
                pass

    # Grapg plotting functions
    def make_plot(self):
        if len(self.val_mae_lst) < 2:
            return
        self.plot_val_mae.removeItem(self.gfx1)
        self.datos = pg.ScatterPlotItem()
        self.gfx1 = self.plot_val_mae.addPlot(title="Absolute MVE")
        self.gfx1.setLabel("left", "loss")
        self.gfx1.setLabel("bottom", "epoch(s)")
        self.datos = self.gfx1.plot(pen="y")
        self.datos.setData(self.val_mae_lst)
        self.gfx1.enableAutoRange(x=True, y=True)

    # Main core 3DCNN model
    def create_3dcnn_regression_model(self):
        model = tf.keras.models.Sequential()

        # Block 1
        model.add(
            tf.keras.layers.Conv3D(
                32,
                kernel_size=(3, 3, 3),
                activation="relu",
                padding="same",
                input_shape=self.train_data.shape[1:],
            )
        )
        model.add(BatchNormalization())
        model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2)))

        # Block 2
        model.add(
            tf.keras.layers.Conv3D(64, kernel_size=(3, 3, 3), activation="relu", padding="same")
        )
        model.add(tf.keras.layers.MaxPooling3D(pool_size=(1, 2, 2)))

        # Block 3
        model.add(
            tf.keras.layers.Conv3D(128, kernel_size=(3, 3, 3), activation="relu", padding="same")
        )
        model.add(tf.keras.layers.MaxPooling3D(pool_size=(2, 2, 2)))

        # Block 4
        model.add(
            tf.keras.layers.Conv3D(256, kernel_size=(3, 3, 3), activation="relu", padding="same")
        )
        model.add(GlobalAveragePooling3D())  # Global Average Pooling instead of Flatten

        # Fully Connected Layers
        model.add(tf.keras.layers.Dense(128, activation=LeakyReLU(alpha=0.01)))
        # model.add(tf.keras.layers.Dropout(0.01))  # 0.1 # 0.3
        model.add(tf.keras.layers.Dense(64, activation=LeakyReLU(alpha=0.01)))
        # model.add(tf.keras.layers.Dropout(0.01)) #0.5  #0.1 # 0.3 # remove this
        model.add(tf.keras.layers.Dense(1, activation=self.activation_func))

        return model

    # the batch K-fold run code
    def confrim_train_3cnn_kfold(self, cmd_mode=False):
        self.post_train_evalulation = True  # must set TRue for operating in k-fold post evaluation

        mode_msg = (
            "\nNOTE: Single dataset mode is on. Only current dataset will be trained"
            if self.train_singlemode_flag.isChecked()
            else "\nNOTE: Multi-dataset mode is on. All datafiles will be trained."
        )
        mode_msg = (
            mode_msg
            + "\nNOTE: Ensure the number of rotation and pixel resoultion to match training | inference dataset"
        )

        if (
            self.qm.question(
                self,
                "CDCCS",
                f"Start K-fold cross validation 3DCCS model?\nNOTE: Train/Test/Validation ratios will be autoset.{mode_msg}",
                self.qm.Yes | self.qm.No,
            )
            == self.qm.No
            and cmd_mode == False
        ):
            return
        self.run_mode = "Train 3DCNN model K-fold mode"
        self.update_gui_vars()
        # ============================================================== # loop through all files datasets
        self.cur_train_res = self.img_dim  # 32 # strat with optimium pixels
        # ==============================================================
        current_folder = os.path.join(cur_path, "datasets")
        if self.train_singlemode_flag.isChecked():
            print(
                colored(
                    f"\nTraining single dataset mode for:\nSMILE datafile: {self.csv_file_path}\n2D projections: {self.dataset_path}",
                    "yellow",
                )
            )
            csv_files = [self.csv_file_path]  # just a single file mode
        else:
            print(
                colored(
                    f"\nTraining batch mode for all datafiles in path: {cur_path}\\datasets",
                    "yellow",
                )
            )
            # List all files in the directory and filter for CSV files
            csv_files = [file for file in os.listdir(current_folder) if file.endswith(".csv")]
            print(
                colored(
                    f"\nTotal of {len(csv_files)} training datafile(s) found batching running Training operation",
                    "green",
                )
            )
            for itm in csv_files:
                print(colored(os.path.join(current_folder, itm), "white"))

        for cur_file in csv_files:
            print(
                colored(
                    f"\n#---------------------------------------------------------------------",
                    "yellow",
                )
            )
            cur_file = (
                os.path.join(current_folder, cur_file)
                if self.train_singlemode_flag.isChecked() == False
                else cur_file
            )  # for running training
            self.train_msdata.setText(cur_file)
            self.raw_smile_datafile.setText(
                get_pathname_without_ext(cur_file) + ".csv"
            )  # for inference & filter results
            self.train_2d_projection_dirpath.setText(
                os.path.join(
                    get_pathname_without_ext(cur_file) + "_optimized_structure", "2D_projections"
                )
            )
            print(colored(f"\nCurrently training on : {cur_file}", "white"))
            # ==============================================================

            if self.use_multipixel_flag:
                try:
                    multi_image_dims = [
                        int(pxl) for pxl in self.mult_pixel.toPlainText().strip().split(",")
                    ]
                    print(
                        colored(
                            f"\n Batch processing for image dimension(s): {multi_image_dims}",
                            "green",
                        )
                    )
                except:
                    print(
                        colored(
                            f"\n Multi pixels mode if turned off. Using SINGLE image dimension {self.img_dim}",
                            "red",
                        )
                    )
                    multi_image_dims = [self.img_dim]
            else:
                print(
                    colored(
                        f"\nCurrently training single pixel resoultion : {self.img_dim}x{self.img_dim} pixels",
                        "white",
                    )
                )
                multi_image_dims = [self.img_dim]

            for (
                mol_res
            ) in (
                multi_image_dims
            ):  # [8,64,128,16]:                                                  # Train for different resoultion of the 2D image in pixelx x pixels

                self.image_dim_slider.setValue(mol_res)
                print(colored(f"\nTraining for pixel(s) : {mol_res} pixels", "white"))
                self.cur_train_res = mol_res  # set the vfal for cur mol resoultion3
                for cur_run in range(self.exp_counter):  # run for each resoultion 3 times
                    print(
                        colored(
                            f"\n#---------------| Running simulation No#.{cur_run+1}/5:\n", "yellow"
                        )
                    )
                    self.random_seed_slider.setValue(random.randint(0, 99999))  # 1985
                    set_random_seed(self.random_seed_slider.value())
                    self.clear_gpu_mem()  # clear gpu memory
                    self.select_gpu(init=False)
                    self.configure_split_sliders()
                    self.update_gui_vars()
                    self.configure_split_sliders()
                    # ===========Load data
                    self.console_show_store_vars()  # finally shwo console variables
                    self.start_time = time.time()
                    self.run_mode = "K-Fold Cross-validation training mode"
                    self.set_gpu_memory_growth(
                        gpu_index=self.gpu_index, set_flag=self.set_gpu_growth.isChecked()
                    )  # set GPU mem growth
                    self.alternate_process_data_and_models()  # for k-fold training k-fold

    # this is the batch training methods I currently used to do all training in batch. need manual customization at moment to change values
    def confrim_train_3cnn(self, cmd_mode=False):

        self.post_train_evalulation = True  # must set TRue for operating in k-fold post eva

        mode_msg = (
            "\nNOTE: Single dataset mode is on. Only current dataset will be trained"
            if self.train_singlemode_flag.isChecked()
            else "\nNOTE: Multi-dataset mode is on. All datafiles will be trained"
        )
        mode_msg = (
            mode_msg
            + "\nNOTE: Ensure the number of rotation and pixel resoultion to match training | inference dataset"
        )
        if (
            self.qm.question(
                self, "CDCCS", f"Start train 3DCCS model?{mode_msg}", self.qm.Yes | self.qm.No
            )
            == self.qm.No
            and cmd_mode == False
        ):
            return

        self.run_mode = "3DCNN model training in randomized sampling mode "
        # Make sure the split sliders can go from 0 to 100 before reading their values.
        self.configure_split_sliders()
        self.update_gui_vars()
        # Re-apply after update_gui_vars() in case GUI/config logic changes slider limits.
        self.configure_split_sliders()

        if (
            sum(
                [
                    self.train_ratio_slider.value(),
                    self.test_ratio_slider.value(),
                    self.val_ratio_slider.value(),
                ]
            )
            > 100
        ):
            self.qm.critical(
                self,
                "Error!",
                "The sum of Train,Test, Validation samples cannot exceed 100%!\n Enter valid percentage values to continue",
            )
            return
        # ============================================================== # ==========================
        self.cur_train_res = 32
        # ============================================================== # loop through all files datasets
        self.post_train_evalulation = (
            True  # for inference on validatio nset afetr training (automatically)
        )
        current_folder = os.path.join(cur_path, "datasets")
        if self.train_singlemode_flag.isChecked():
            print(
                colored(
                    f"\nTraining single dataset mode for:\nSMILE datafile: {self.csv_file_path} \n2D projections: {self.dataset_path}",
                    "yellow",
                )
            )
            csv_files = [self.csv_file_path]  # just a single file mode
        else:
            print(
                colored(
                    f"\nTraining batch mode for all datafiles in path: {cur_path}\\datasets",
                    "yellow",
                )
            )
            # List all files in the directory and filter for CSV files
            csv_files = [file for file in os.listdir(current_folder) if file.endswith(".csv")]
            print(
                colored(
                    f"\nTotal of {len(csv_files)} training datafile(s) found batching running Training operation",
                    "green",
                )
            )
            for itm in csv_files:
                print(colored(os.path.join(current_folder, itm), "white"))

        for cur_file in csv_files:
            print(
                colored(
                    f"\n#---------------------------------------------------------------------",
                    "yellow",
                )
            )
            cur_file = (
                os.path.join(current_folder, cur_file)
                if self.train_singlemode_flag.isChecked() == False
                else cur_file
            )  # for running training
            self.train_msdata.setText(cur_file)
            self.raw_smile_datafile.setText(
                get_pathname_without_ext(cur_file) + ".csv"
            )  # for inference & filter results
            self.train_2d_projection_dirpath.setText(
                os.path.join(
                    get_pathname_without_ext(cur_file) + "_optimized_structure", "2D_projections"
                )
            )
            print(colored(f"\nCurrently training on : {cur_file}", "white"))
            # ============================================================== #======@###########################################################################################################
            if self.use_multipixel_flag:
                try:
                    multi_image_dims = [
                        int(pxl) for pxl in self.mult_pixel.toPlainText().strip().split(",")
                    ]
                    print(
                        colored(
                            f"\nBatch processing for image dimension(s): {multi_image_dims}",
                            "green",
                        )
                    )
                except:
                    print(
                        colored(
                            f"\nMulti pixels not found. Using single image dimension {self.img_dim}",
                            "red",
                        )
                    )
                    multi_image_dims = [self.img_dim]
            else:
                print(
                    colored(
                        f"\nCurrently training single pixel resoultion : {self.img_dim}x{self.img_dim} pixels",
                        "white",
                    )
                )
                multi_image_dims = [self.img_dim]

            for (
                mol_res
            ) in (
                multi_image_dims
            ):  # [8,64,128,16]:                                                  # Train for different resoultion of the 2D image in pixelx x pixels

                self.image_dim_slider.setValue(mol_res)
                print(
                    colored(
                        f"\nCurrently training resoultion : {mol_res} x {mol_res} pixels", "white"
                    )
                )
                self.cur_train_res = mol_res  # set the vfal for cur mol resoultion3
                for cur_run in range(self.exp_counter):  # run for each resoultion 3 times
                    print(
                        colored(
                            f"\n#---------------| Running simulation No#.{cur_run+1}/5:\n", "yellow"
                        )
                    )
                    self.random_seed_slider.setValue(random.randint(0, 99999))  # 1985
                    set_random_seed(self.random_seed_slider.value())
                    self.clear_gpu_mem()
                    self.select_gpu(init=False)
                    self.configure_split_sliders()
                    self.update_gui_vars()
                    self.configure_split_sliders()
                    self.console_show_store_vars()  # show console variable
                    # ===========Load data
                    self.start_time = time.time()
                    self.run_mode = "Loading 3DCNN training data"
                    # clear gpu memory
                    self.set_gpu_memory_growth(
                        gpu_index=self.gpu_index, set_flag=self.set_gpu_growth.isChecked()
                    )  # set GPU mem growth
                    self.process_data_and_models()  # creates the shuffle data based on random seed and compiles the model to fit input data
                    # =========== Start real training
                    self.start_time = time.time()
                    self.run_mode = "Train 3DCNN model"
                    self.train_3dcnn()

    # Function to load and resize images from each component folder (x, y, z) for a single sample
    def load_sample_images(self, sample_path):
        # print("Loading images from : %s"%sample_path)
        x_images = self.load_images_from_folder(os.path.join(sample_path, "x"))
        y_images = self.load_images_from_folder(os.path.join(sample_path, "y"))
        z_images = self.load_images_from_folder(os.path.join(sample_path, "z"))

        return np.concatenate(
            [x_images, y_images, z_images], axis=0
        )  # todo: concatianert the gtration values here

    # Function to load and resize images from each component folder (x, y, z) using 2-bit PNg image
    def load_images_from_folder(self, folder_path):
        images = []
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path)
            gray_image = img.convert("L")  # Convert the image to grayscale (just in case)
            gray_image = gray_image.resize(
                (self.img_dim, self.img_dim)
            )  # resize the grayscaled image
            image_array = np.array(gray_image)  # convert image to numpy image array
            two_bit_array = (image_array > self.image_threshold).astype(
                np.uint8
            )  # use thresthold to convert to 0 & 1 (2 bit) for each pixel (2 bit from 256 bits)
            two_bit_image = Image.fromarray(
                two_bit_array * 255
            )  # Convert the two-bit array back to a PIL image #Scaling 0 and 1 to 0 and 255
            img = np.array(two_bit_image)  # convert to numpy array
            img = img[:, :, np.newaxis]
            images.append(img)  # Append 2-bit numpy image to list

        return np.array(images)

    # Function to read folders and match with CSV                                    # ths is the function to managet hemissing exp_ccs case

    def read_folders_and_match_csv(self, parent_directory, csv_file_path):

        # Declare variables for CCS and metadata.
        name_to_exp_ccs = {}
        name_to_extract_mass = {}
        name_to_mz_ratio = {}
        name_to_mol_name = {}
        name_to_smiles = {}
        csv_key_order = []

        # These lists are now populated in load_data() in the same order as found_list.
        # This fixes the previous name/SMILES mismatch problem during inference.
        self.mol_name_list = []
        self.mol_smiles_list = []
        self.found_sample_names = []

        self.exp_ccs_is_missing = False

        print(colored("#Reading database..."))

        if self.inf_adduct_type == "N/A":
            print(
                colored(
                    "\n[WARNING] Adduct type unknown for INFERENCE. Setting inference default to:",
                    "red",
                ),
                colored("{[M-H]- Monoisotopic masses ", "white"),
            )
        else:
            print(
                colored("\nSetting current adduct type for INFERENCE to :", "green"),
                colored(f"{self.inf_adduct_type}", "white"),
            )

        if self.train_adduct_type == "N/A":
            print(
                colored(
                    "\n[WARNING] Adduct type unknown for TRAINING. Setting inference default to:",
                    "red",
                ),
                colored("[M-H]- Monoisotopic masses ", "white"),
            )
        else:
            print(
                colored("\nSetting current adduct type for TRAINING to  :", "green"),
                colored(f"{self.train_adduct_type}", "white"),
            )

        with open(csv_file_path, "r", newline="", encoding="latin-1") as csvfile:
            reader = csv.DictReader(csvfile)
            count = 0
            skipped = 0

            print(
                colored(
                    f"\n#Using computed molecular mass from SMILES: {self.use_computed_mass}",
                    color="green",
                )
            )

            for row in tqdm(reader, desc="Reading database :", ncols=200):

                row_id = str(row.get("ID", "")).strip()
                row_name = str(row.get("name", row_id)).strip()
                row_smiles = str(row.get("SMILES", "")).strip()

                if row_id == "" and row_name == "":
                    skipped += 1
                    continue

                # Projection folders are named by ID when AllCCSID_name_flag is True;
                # otherwise they are named by molecule name.
                name = row_id.lower() if self.AllCCSID_name_flag else row_name.lower()

                try:
                    if self.use_computed_mass:
                        mol_mass = self.compute_exact_mass(row_smiles)
                    else:
                        mol_mass = float(row["extract_mass"])
                except Exception:
                    skipped += 1
                    continue

                if (mol_mass < self.max_cutoff_weight) and (mol_mass > self.min_cutoff_weight):

                    raw_exp_ccs = row.get("exp_ccs", 0)

                    if self.autoset_expccs_flag and self.post_train_evalulation == False:
                        try:
                            exp_ccs = float(raw_exp_ccs) if raw_exp_ccs not in ("", None) else 0.0
                        except ValueError:
                            self.exp_ccs_is_missing = True
                            exp_ccs = 0.0
                    else:
                        try:
                            exp_ccs = float(row["exp_ccs"])
                        except Exception:
                            exp_ccs = float(row["CCS"])

                    name_to_exp_ccs[name] = round(exp_ccs, self.set_precision)
                    name_to_extract_mass[name] = round(mol_mass, self.set_precision)
                    name_to_mol_name[name] = row_name
                    name_to_smiles[name] = row_smiles
                    csv_key_order.append(name)

                    try:
                        name_to_mz_ratio[name] = round(float(row["mz_ratio"]), self.set_precision)
                    except Exception:
                        if self.post_train_evalulation == True:
                            adduct = (
                                "[M-H]-" if self.inf_adduct_type == "N/A" else self.inf_adduct_type
                            )
                        else:
                            adduct = (
                                "[M-H]-"
                                if self.train_adduct_type == "N/A"
                                else self.train_adduct_type
                            )
                        name_to_mz_ratio[name] = self.compute_mz_ratio(row_smiles, adduct)

                    count += 1
                else:
                    skipped += 1

            print(colored("________________________________________________________", "white"))
            print(colored("#Total Molecules added  : ", "green"), count)
            print(colored("#Total Molecules skipped: ", "red"), skipped)
            print(colored("________________________________________________________", "white"))

        # Match CSV rows to actual projection folders. Keep CSV order to keep metadata aligned.
        projection_parent = parent_directory
        if not os.path.isabs(projection_parent):
            projection_parent = os.path.join(cur_path, projection_parent)

        folder_name_lookup = {
            folder.lower(): folder
            for folder in os.listdir(projection_parent)
            if os.path.isdir(os.path.join(projection_parent, folder))
        }

        found_list = []
        missing_list = []
        name_to_folder_name = {}

        seen = set()
        for key in csv_key_order:
            if key in seen:
                continue
            seen.add(key)
            if key in folder_name_lookup:
                found_list.append(key)
                name_to_folder_name[key] = folder_name_lookup[key]
            else:
                missing_list.append(key)

        print("\n#Sample folders with CCS: %d | Missing: %d" % (len(found_list), len(missing_list)))

        return (
            found_list,
            name_to_exp_ccs,
            name_to_extract_mass,
            name_to_mz_ratio,
            name_to_mol_name,
            name_to_smiles,
            name_to_folder_name,
        )

    def load_data(self, dataset_path, csv_file_path):
        (
            found_list,
            name_to_exp_ccs,
            name_to_extract_mass,
            name_to_mz_ratio,
            name_to_mol_name,
            name_to_smiles,
            name_to_folder_name,
        ) = self.read_folders_and_match_csv(dataset_path, csv_file_path)

        samples = []
        exp_ccs_list = []
        exp_extract_mass_list = []
        exp_mz_ratio_list = []

        # Reset metadata lists every time data are loaded.
        # They are appended in exactly the same order as samples.
        self.found_sample_names = []
        self.mol_name_list = []
        self.mol_smiles_list = []

        if self.sample_limit is not None:
            print(
                colored(
                    "WARNING! Only first %d samples from data will be used for model training & validation"
                    % self.sample_limit,
                    "yellow",
                )
            )
            found_list = found_list[: self.sample_limit]

        count = 0
        for sample_key in tqdm(found_list, desc="#Scanning data :", ncols=100):
            actual_folder_name = name_to_folder_name.get(sample_key, sample_key)
            sample_path = os.path.join(dataset_path, actual_folder_name)

            exp_ccs_list.append(name_to_exp_ccs[sample_key])
            exp_extract_mass_list.append(name_to_extract_mass[sample_key])
            exp_mz_ratio_list.append(name_to_mz_ratio[sample_key])

            sample_data = self.load_sample_images(sample_path)
            samples.append(sample_data)

            # These lists are now correctly aligned with samples, exp_ccs, and predictions.
            self.found_sample_names.append(actual_folder_name)
            self.mol_name_list.append(name_to_mol_name.get(sample_key, actual_folder_name))
            self.mol_smiles_list.append(name_to_smiles.get(sample_key, ""))

            count += 1
            self.update_progressBar(len(found_list), count)

        data = np.array(samples)
        exp_ccs = np.array(exp_ccs_list)
        exp_extract_mass = np.array(exp_extract_mass_list)
        exp_mz_ratio = np.array(exp_mz_ratio_list)

        return data, exp_ccs, exp_extract_mass, exp_mz_ratio

    def alternate_process_data_and_models(self):
        """
        Perform simple K-Fold cross-validation (no nested CV).
        Trains and evaluates a 3DCNN regression model on each fold,
        computes metrics, and saves:
          - Per-fold results into a single CSV file
          - Final aggregated summary into a JSON file
        """
        # Start
        self.start_time = time.time()
        self.run_mode = "Train 3DCNN model K-Fold mode"

        self.dataset_path = self.train_2d_projection_dirpath.toPlainText()
        self.csv_file_path = self.train_msdata.toPlainText()
        # Load data
        self.data, self.exp_ccs, self.exp_extract_mass, self.exp_mz_ratio = self.load_data(
            self.dataset_path, self.csv_file_path
        )

        # Shuffle data
        set_random_seed(self.random_seed)
        self.shuffled_indices = list(range(len(self.data)))
        random.shuffle(self.shuffled_indices)

        # Shuffle everything consistently
        self.data = self.data[self.shuffled_indices]
        self.exp_ccs = self.exp_ccs[self.shuffled_indices]
        self.exp_extract_mass = [self.exp_extract_mass[i] for i in self.shuffled_indices]
        self.exp_mz_ratio = [self.exp_mz_ratio[i] for i in self.shuffled_indices]
        self.found_sample_names = [self.found_sample_names[i] for i in self.shuffled_indices]
        self.mol_name_list = [self.mol_name_list[i] for i in self.shuffled_indices]
        self.mol_smiles_list = [self.mol_smiles_list[i] for i in self.shuffled_indices]

        # Convert to np arrays
        self.exp_extract_mass = np.array(self.exp_extract_mass)
        self.exp_mz_ratio = np.array(self.exp_mz_ratio)

        # Normalize features
        self.extract_mass_normalized = (
            (self.exp_extract_mass - self.exp_extract_mass.min())
            / (self.exp_extract_mass.max() - self.exp_extract_mass.min())
        ) * 256
        self.mz_ratio_normalized = (
            (self.exp_mz_ratio - self.exp_mz_ratio.min())
            / (self.exp_mz_ratio.max() - self.exp_mz_ratio.min())
        ) * 256

        # print("\n#Data shape before feature addition:", self.data.shape)

        self.image_data = self.data
        self.input_data = np.empty(
            (
                self.image_data.shape[0],
                self.image_data.shape[1],
                self.image_data.shape[2],
                self.image_data.shape[3],
                3,
            )
        )
        self.input_data[:, :, :, :, 0] = self.image_data[:, :, :, :, 0]
        self.input_data[:, :, :, :, 1] = (
            self.mz_ratio_normalized[:, np.newaxis, np.newaxis, np.newaxis] * 0
        )  # ,ade all mz_ratio values zero
        self.data = self.input_data

        print("#Data shape after feature addition :", self.data.shape)

        # Prepare storage
        results = []

        # Define K-Fold
        n_splits = 4  # this is fixed
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=abs(self.random_seed))

        # Loop folds
        fold = 1
        self.val_index = []

        for train_idx, val_idx in kf.split(self.data):
            print(f"\nProcessing Fold : {fold}")
            self.val_idx = val_idx
            # print("K-fold validation indexes: ", self.val_idx)

            # Split
            X_train, X_val = self.data[train_idx], self.data[val_idx]
            y_train, y_val = self.exp_ccs[train_idx], self.exp_ccs[val_idx]

            # Build model
            self.train_data = X_train
            self.regression_model = self.create_3dcnn_regression_model()
            if self.loss_func == "Huber":
                model_loss_func = Huber(delta=1.0)
            else:
                model_loss_func = "mse"

            self.regression_model.compile(
                optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),
                loss=model_loss_func,
                metrics=[
                    "mean_squared_error",
                    "mean_absolute_error",
                    "mean_absolute_percentage_error",
                ],
            )

            # display model summry
            if self.model_summary_flag:
                print(
                    colored(
                        "Model summary:\n------------------------------------------------\n",
                        "green",
                    )
                )
                self.regression_model.summary()
                print(colored("\n------------------------------------------------\n", "green"))

            # Train
            self.history = self.regression_model.fit(
                X_train,
                y_train,
                epochs=self.train_epoch,
                batch_size=self.batch_size,
                validation_data=(X_val, y_val),
                verbose=self.verbosity,
            )

            # Evaluate
            val_scores = self.regression_model.evaluate(X_val, y_val, verbose=1)
            predictions = self.regression_model.predict(X_val).flatten()

            slope, intercept, r_value, p_value, std_err = linregress(y_val, predictions)
            mae = mean_absolute_error(y_val, predictions)
            perc_std_err, std_dev, skipped_count, skipped_pct = percentage_std_error(
                y_val, predictions
            )
            pearson_corr, _ = pearsonr(y_val, predictions)

            # Save fold results
            fold_result = {
                "Fold": fold,
                "Dataset_ID": self.dataset_id,
                "Validation_Loss": val_scores[0],
                "Validation_Accuracy": val_scores[1],
                "Pearson_Corr": pearson_corr,
                "Slope": slope,
                "Intercept": intercept,
                "R_squared": r_value**2,
                "P_value": p_value,
                "MAE": mae,
                "Std_Error": std_err,
                "Relative_Percent_Error": perc_std_err,
                "Std_Deviation": std_dev,
                "Skipped_Count": skipped_count,
                "Skipped_Pct": skipped_pct,
            }
            results.append(fold_result)

            # Print fold results
            print(colored("___________________________________________________", "green"))
            for k, v in fold_result.items():
                print(f"{k:30}: {v}")
            print(colored("___________________________________________________", "green"))

            # conduct inference
            self.inference_3dccs_kfold(
                post_train_evalulation=True, kfold=f"_kfold={fold}_"
            )  # for k-fold here
            self.clear_gpu_mem()
            fold += 1

        os.makedirs(self.evaluations_dir, exist_ok=True)

        # Save all fold results into one CSV
        df_results = pd.DataFrame(results)
        csv_path = os.path.join(self.evaluations_dir, f"kfold_results.csv")
        df_results.to_csv(csv_path, index=False)
        print(f"\nSaved per-fold results to {csv_path}")

        # Compute summary
        def summary(metric_list):
            return np.mean(metric_list), np.std(metric_list)

        summary_dict = {}
        for col in df_results.columns:
            if col in ["Fold", "Dataset_ID"]:
                continue
            mean, std = summary(df_results[col].values)
            summary_dict[col] = {"mean": float(mean), "std": float(std)}

        # Save summary to JSON
        json_path = os.path.join(self.evaluations_dir, f"kfold_summary.json")
        with open(json_path, "w") as f:
            json.dump(summary_dict, f, indent=4)
        print(f"Saved summary to {json_path}")

        self.clear_gpu_mem()

    def process_data_and_models(self):

        training_data_processing_start_perf = time.perf_counter()
        self.run_mode = "Train 3DCNN model randomized mode"

        # Load the data and experimental CCS values
        self.data, self.exp_ccs, self.exp_extract_mass, self.exp_mz_ratio = self.load_data(
            self.dataset_path, self.csv_file_path
        )

        # ============================================================ Shuffle the data based on seed
        set_random_seed(self.random_seed)  # shuffle based on random seed

        self.shuffled_indices = list(
            range(len(self.data))
        )  # Get the random list of indices corrpsondin gt o length of self.data
        random.shuffle(self.shuffled_indices)

        # ====================================================================================

        self.data = self.data[self.shuffled_indices]  # assign new suffled data ( projections 2D)
        self.exp_ccs = self.exp_ccs[
            self.shuffled_indices
        ]  # assign corrsponding shuffled ccs values
        self.exp_extract_mass = [
            self.exp_extract_mass[i] for i in self.shuffled_indices
        ]  # for exact mass associted values for corspondingshuffled values
        self.exp_mz_ratio = [
            self.exp_mz_ratio[i] for i in self.shuffled_indices
        ]  # for the exp_mz_ratio associated value for corspondingshuffled values
        self.found_sample_names = [
            self.found_sample_names[i] for i in self.shuffled_indices
        ]  # shuffle foldernames  based on same suffle index (use it becuase found_sample)names is already a list
        self.mol_name_list = [
            self.mol_name_list[i] for i in self.shuffled_indices
        ]  # holds original name of molecule ion
        self.mol_smiles_list = [self.mol_smiles_list[i] for i in self.shuffled_indices]
        # ========================

        # Convert lists to numpy arrays
        self.exp_extract_mass = np.array(self.exp_extract_mass)
        self.exp_mz_ratio = np.array(self.exp_mz_ratio)

        # Normalize additional features to 8-bit range (0-256)
        self.extract_mass_normalized = (
            (self.exp_extract_mass - self.exp_extract_mass.min())
            / (self.exp_extract_mass.max() - self.exp_extract_mass.min())
        ) * 256
        self.mz_ratio_normalized = (
            (self.exp_mz_ratio - self.exp_mz_ratio.min())
            / (self.exp_mz_ratio.max() - self.exp_mz_ratio.min())
        ) * 256

        # print("\n#Data shape before extract_mass & mz_ratio:", self.data.shape)

        self.image_data = self.data  # (shape: (148, 15, 128, 128, 1))

        # Create input data with additional features
        self.input_data = np.empty(
            (
                self.image_data.shape[0],
                self.image_data.shape[1],
                self.image_data.shape[2],
                self.image_data.shape[3],
                3,
            )
        )
        self.input_data[:, :, :, :, 0] = self.image_data[:, :, :, :, 0]  # Image data
        self.input_data[:, :, :, :, 1] = (
            self.mz_ratio_normalized[:, np.newaxis, np.newaxis, np.newaxis] * 0
        )  # mz_ratio the extract_mass is now removed to 2 become 1 dim

        self.data = self.input_data  # update new shape

        # print("#Data shape after extract_mass & mz_ratio :", self.data.shape)

        # ===============================================================================SAMPLE SPLITTING
        # Data saturation mode:
        # Train slider      = training percentage
        # Test slider       = internal validation percentage used during model.fit()
        # Validation slider = internal test percentage used for post-training evaluation
        # Any remaining samples are intentionally unused.
        self.num_samples = self.data.shape[
            0
        ]  # Assuming all samples have the same number of components

        # Split the data into training, validation, test, and unused sets
        # The debug print confirms the actual slider values being used at runtime.
        print(
            colored(
                f"DEBUG raw GUI slider values | "
                f"Train = {self.train_ratio_slider.value()}, "
                f"Test = {self.test_ratio_slider.value()}, "
                f"Validation = {self.val_ratio_slider.value()}",
                "yellow",
            )
        )

        self.train_percent = round(
            self.train_ratio_slider.value() / 100, 2
        )  # e.g., 0.20, 0.40, 0.60
        self.val_percent = round(
            self.test_ratio_slider.value() / 100, 2
        )  # internal validation during training
        self.test_percent = round(
            self.val_ratio_slider.value() / 100, 2
        )  # internal post-training test

        self.used_percent = self.train_percent + self.val_percent + self.test_percent
        self.unused_percent = round(1.0 - self.used_percent, 2)

        if self.used_percent > 1.0:
            self.qm.critical(
                self, "Error!", "Train + Test + Validation percentages must not exceed 100%."
            )
            return

        self.num_train_samples = int(self.train_percent * self.num_samples)
        self.num_val_samples = int(self.val_percent * self.num_samples)
        self.num_test_samples = int(self.test_percent * self.num_samples)

        train_start = 0
        train_end = self.num_train_samples

        val_start = train_end
        val_end = val_start + self.num_val_samples

        test_start = val_end
        test_end = test_start + self.num_test_samples

        # ============== divide the shuffled self.data based on train, validation, test, and unused samples
        self.train_data = self.data[train_start:train_end]  # Experimental 2D projections
        self.train_exp_ccs = self.exp_ccs[train_start:train_end]  # CCS values from experiment
        self.train_exp_mz_ratio = self.exp_mz_ratio[
            train_start:train_end
        ]  # Train experimental mz_ratio values

        self.val_data = self.data[val_start:val_end]
        self.val_exp_ccs = self.exp_ccs[val_start:val_end]
        self.val_exp_mz_ratio = self.exp_mz_ratio[val_start:val_end]

        self.test_data = self.data[test_start:test_end]
        self.test_exp_ccs = self.exp_ccs[test_start:test_end]
        self.test_exp_mz_ratio = self.exp_mz_ratio[test_start:test_end]

        # Optional: keep unused data in memory for checking only
        self.unused_data = self.data[test_end:]
        self.unused_exp_ccs = self.exp_ccs[test_end:]

        # ==============================================================================

        print(
            colored(
                f"Train: Test: Validation: Unused ratio | "
                f"{self.train_percent} : {self.val_percent} : {self.test_percent} : {self.unused_percent}",
                "green",
            )
        )

        print(
            colored(
                f"Samples | Train: {len(self.train_data)}, "
                f"Internal validation: {len(self.val_data)}, "
                f"Internal test: {len(self.test_data)}, "
                f"Unused: {len(self.unused_data)}",
                "cyan",
            )
        )

        # ==============================================================================

        # Create the regression model
        self.regression_model = self.create_3dcnn_regression_model()

        if self.loss_func == "Huber":
            model_loss_func = Huber(delta=1.0)
        if self.loss_func == "MSE":
            model_loss_func = "mse"
        # Compile the model
        self.regression_model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),  # Import Huber Loss
            loss=model_loss_func,
            metrics=[
                "mean_squared_error",  # loss='mean_squared_error'
                "mean_absolute_error",
                "mean_absolute_percentage_error",
            ],
        )

        self.training_data_processing_elapsed_min = (
            time.perf_counter() - training_data_processing_start_perf
        ) / 60.0
        print(
            colored(
                f"Training data loading/splitting/model-compilation time: {self.training_data_processing_elapsed_min:.4f} min",
                "green",
            )
        )

    # ================================================================================================ Processing for inference

    def process_data_for_inference(self):
        if (
            self.dataset_path.strip() == ""
            or self.csv_file_path.strip() == ""
            or self.reg_model_fname.strip() == ""
        ):
            self.qm.critical(
                self,
                "Error!",
                "Please input the SMIEL datafile, corrsonding 2D projection folder and trained model path ",
            )
            return

        print(colored("#Processing data for inference...", "yellow"))

        # Load the data and experimental CCS values
        self.data, self.exp_ccs, self.exp_extract_mass, self.exp_mz_ratio = self.load_data(
            self.dataset_path, self.csv_file_path
        )  # for now put same

        """Do not shuffle 'anything'
        #============================================================ Shuffle the data based on seed
        set_random_seed(self.random_seed)   
        self.shuffled_indices = list(range(len(self.data)))                                      # Get the random list of indices corrpsondin gt o length of self.data
        #====================================================================================    # Shuffle indices (optional here )
        random.shuffle(self.shuffled_indices)                                                   # do not need shuffling for Inefrence just do as it as
        self.data               = self.data[self.shuffled_indices]                               # assign new suffled data ( projections 2D)
        self.exp_ccs            = self.exp_ccs[self.shuffled_indices]                            # assign corrsponding shuffled ccs values 
        self.exp_extract_mass   = [self.exp_extract_mass[i]   for i in self.shuffled_indices]    # for exact mass associted values for corspondingshuffled values 
        self.exp_mz_ratio       = [self.exp_mz_ratio[i]       for i in self.shuffled_indices]    # for the exp_mz_ratio associated value for corspondingshuffled values 
        self.found_sample_names = [self.found_sample_names[i] for i in self.shuffled_indices]    # foudn sampel names index
        self.mol_name_list      = [self.mol_name_list[i]      for i in self.shuffled_indices]    # holds original name of molecule ion
        self.mol_smiles_list    = [self.mol_smiles_list[i]    for i in self.shuffled_indices]
        #========================
        """

        # Convert lists to numpy arrays
        self.exp_extract_mass = np.array(self.exp_extract_mass)
        self.exp_mz_ratio = np.array(self.exp_mz_ratio)

        # Normalize additional features to 8-bit range (0-256)
        self.extract_mass_normalized = (
            (self.exp_extract_mass - self.exp_extract_mass.min())
            / (self.exp_extract_mass.max() - self.exp_extract_mass.min())
        ) * 256
        self.mz_ratio_normalized = (
            (self.exp_mz_ratio - self.exp_mz_ratio.min())
            / (self.exp_mz_ratio.max() - self.exp_mz_ratio.min())
        ) * 256

        # print("\n#Data shape before extract_mass & mz_ratio:", self.data.shape)

        self.image_data = self.data  # (shape: (148, 15, 128, 128, 1))

        # Create input data with additional features
        self.input_data = np.empty(
            (
                self.image_data.shape[0],
                self.image_data.shape[1],
                self.image_data.shape[2],
                self.image_data.shape[3],
                3,
            )
        )
        self.input_data[:, :, :, :, 0] = self.image_data[:, :, :, :, 0]  # Image data
        # self.input_data[:, :, :, :, 1] = self.extract_mass_normalized[:, np.newaxis, np.newaxis, np.newaxis] # Extract_mass
        self.input_data[:, :, :, :, 1] = (
            self.mz_ratio_normalized[:, np.newaxis, np.newaxis, np.newaxis] * 0
        )  # mz_ratio  , originally #2
        self.data = self.input_data  # update new shape
        self.num_train_samples = 0  # since we only use ext. test for all
        self.num_val_samples = 0  # since we only use ext.  test for all

        print("#Data shape after extract_mass & mz_ratio :", self.data.shape)

        # =========================================================== No sample splitting
        self.num_samples = self.data.shape[0]
        self.test_data = self.data
        self.test_exp_ccs = self.exp_ccs

    # =============================================  cut if posblem

    def save_config(self, cfg_fpath):
        # Create a dictionary to store all relevant variables
        state = {
            "dataset_id": self.dataset_id,
            "last_epoch": self.last_epoch,
            "history_loss": self.history_loss,
            "history_mse": self.history_mse,
            "mean_val_abs_error": self.mean_val_abs_error,
            "history_mean_val_abs_error": self.history_mean_val_abs_error,
            "history_val_mse": self.history_val_mse,
            "history_val_loss": self.history_val_loss,
            "val_mae_lst": self.val_mae_lst,
            "val_mape_lst": self.val_mape_lst,
            "avgabs_per_err_lst": self.avgabs_per_err_lst,
            "relative_percentage_error_lst": self.relative_percentage_error_lst,
            "std_dev_list": self.std_dev_list,
        }

        os.makedirs(self.evaluations_dir, exist_ok=True)  # Make eval directory if necessary
        # Save the dictionary to a file using pickle
        with open(cfg_fpath, "wb") as f:
            pickle.dump(state, f)

    def load_config(self, cfg_fpath):
        # Load the state dictionary from the file
        with open(cfg_fpath, "rb") as f:
            state = pickle.load(f)

        # Restore variables from the dictionary
        self.dataset_id = state["dataset_id"]
        self.last_epoch = state["last_epoch"]
        self.history_mse = state["history_mse"]
        self.mean_val_abs_error = state["mean_val_abs_error"]
        self.history_val_mse = state["history_val_mse"]
        self.history_mean_val_abs_error = state["history_mean_val_abs_error"]
        self.val_mae_lst = state["val_mae_lst"]
        self.val_mape_lst = state["val_mape_lst"]
        self.avgabs_per_err_lst = state["avgabs_per_err_lst"]
        self.history_loss = state["history_loss"]
        self.history_val_loss = state["history_val_loss"]
        self.relative_percentage_error_lst = state["relative_percentage_error_lst"]
        self.std_dev_list = state["std_dev_list"]

        print(colored(f"#Model trained for {self.last_epoch} epoch(s)", "yellow"))

    # ============================================== remove if problem

    # Define the learning rate schedule function
    def lr_scheduler(self, epoch, lr):
        if (
            (epoch + 1) % (self.lr_decay_epoch)
        ) == 0 and self.use_lr_decay == True:  # Every 100 epochs
            old_lr = lr
            lr = lr * (
                1 - self.lr_decay_frac
            )  # Decay current learning rate by fraction of self.lr_decay_frac
            print(
                colored(
                    f"#Learning rate decay to {round(lr, 8)} from {round(old_lr,8)} i.e. : {self.lr_decay_frac * 100}%  decay @ epoch {epoch}/ {self.lr_decay_epoch}",
                    "green",
                )
            )
            return lr
        return lr

    # The training of 3DCNN models are being here
    def train_3dcnn(self):

        training_wall_start_perf = time.perf_counter()
        training_fit_elapsed_min = np.nan
        self.train_start_time = datetime.datetime.now()  # start time

        # Save the model after every 5 epochs
        self.checkpoint_filepath = os.path.join(
            self.evaluations_dir,
            self.dataset_id
            + f"_{self.img_dim}x{self.img_dim}_rseed{self.random_seed}_regression_model_checkpoint.h5",
        )
        self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_filepath, save_weights_only=False, period=5
        )

        # Load model if it exists, load weights
        if os.path.isfile(self.reg_model_fname):
            print(colored("\n#Previously optimized model found. Continue training...", "yellow"))
            self.regression_model.load_weights(self.reg_model_fname)
            if os.path.isfile(self.model_config):
                print(colored("#Model history found.Loading model losses..."))
                self.load_config(self.model_config)
                self.history_rpe = []

        else:
            print(colored("\nWARNING! No Optimized model found! Starting fresh model", "red"))
            if os.path.isfile(self.Base_3dccs_model) and self.basemodel_flag == True:
                print(colored("#User selected to load model weights from base model", "yellow"))
                self.base_reg_model = self.Base_3dccs_model
                self.regression_model.load_weights(self.base_reg_model)
                print(
                    colored(
                        f"\n#Model weights loaded successfully from base model: {self.Base_3dccs_model}",
                        "yellow",
                    )
                )

            # Initialize history variavle  lists to track history
            self.last_epoch = 0  # previous last epoch trained
            self.history_mse = []
            self.history_val_mse = []
            self.history_mean_val_abs_error = []
            self.val_mape_lst = []
            self.avgabs_per_err_lst = []
            self.history_loss = []
            self.history_val_loss = []
            self.history_rpe = []

            self.relative_percentage_error_lst = []
            self.std_dev_list = []

        # Callback function to track history during training
        class HistoryCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                ccs3d.last_epoch += 1  # add current epoch with last epoch
                ccs3d.history_mse.append(logs["mean_squared_error"])
                ccs3d.history_val_mse.append(logs["val_mean_squared_error"])
                # ccs3d.val_predictions = ccs3d.regression_model.predict(ccs3d.test_data)  # val_data

                ccs3d.mean_val_abs_error = logs[
                    "val_mean_absolute_error"
                ]  #   np.mean(np.abs(test_exp_ccs - val_predictions))                             # mean validation absoulte error
                ccs3d.val_mae_lst.append(logs["val_mean_absolute_error"])  #   append to list
                ccs3d.val_perc_abs_error = logs[
                    "val_mean_absolute_percentage_error"
                ]  #   np.mean(np.abs((test_exp_ccs - val_predictions) / test_exp_ccs)) * 100      # calculate the validation percentage error
                ccs3d.history_mean_val_abs_error.append(ccs3d.mean_val_abs_error)
                ccs3d.val_mape_lst.append(logs["val_mean_absolute_percentage_error"])
                ccs3d.history_loss.append(logs["loss"])  # for train losses
                ccs3d.history_val_loss.append(logs["val_loss"])  # for validation loss

                ccs3d.avgabs_per_err_lst.append(
                    ccs3d.val_perc_abs_error
                )  # (logs['mean_absolute_percentage_error'])
                print(
                    colored(
                        f"\rEpoch {ccs3d.last_epoch}/{ccs3d.train_epoch} - Mean Val. Abs Error : {ccs3d.mean_val_abs_error:.4f} | Mean Perc. Val. Abs Error: {ccs3d.val_perc_abs_error:.2f} %",
                        "yellow",
                    ),
                    end="\r",
                )

                # ============================================================

                if (
                    (epoch + 1) % ccs3d.eval_train_freq == 0 and ccs3d.train_rpe_flag
                ) and ccs3d.test_percent > 0.05:  # temporary modeling  (ignore  thsi if no valdiatio nset)

                    truncate = 1000
                    test_data = ccs3d.test_data[
                        :truncate
                    ]  # Assuming ccs3d.test_data is a NumPy array, truncate
                    num_slices = min(
                        8, len(test_data)
                    )  # Set Number of slices & Determine the size of each slice
                    slice_size = len(test_data) // num_slices
                    predictions = []  # Initialize an empty list to collect predictions

                    print(
                        colored(
                            "-----------------------------------------------------------------------------",
                            "green",
                        )
                    )
                    print(
                        colored(
                            f"\nPredicting for slice# @ epoch: {epoch} |  Test size: {len(test_data)} sample(s)",
                            "green",
                        )
                    )
                    for i in tqdm(range(num_slices)):  # Perform predictions for each slice
                        # print(colored(f"predicting for slice# ", "green") , colored(f"{i}/{num_slices}" , "blue")  ,"\r")
                        start_idx = i * slice_size
                        end_idx = (i + 1) * slice_size if i < num_slices - 1 else len(test_data)
                        slice_predictions = ccs3d.regression_model.predict(
                            test_data[start_idx:end_idx], verbose=1
                        ).flatten()
                        predictions.append(slice_predictions)

                    predictions = np.concatenate(
                        predictions
                    )  # Combine all slices into a single array

                    relative_percentage_error, std_dev, skipped_count, skipped_percentage = (
                        percentage_std_error(ccs3d.test_exp_ccs[:truncate], predictions)
                    )
                    ccs3d.history_rpe.append(relative_percentage_error)
                    print(
                        colored(
                            "-----------------------------------------------------------------------------",
                            "green",
                        )
                    )
                    print(
                        colored(
                            f"Relative Std. % Error     : {relative_percentage_error:.4f} | Skipped : {skipped_percentage}% | Skipped items: {skipped_count}",
                            "green",
                        )
                    )
                    print(colored(f"Std. deviation %          : {std_dev:.4f}", "blue"))

                    ccs3d.relative_percentage_error_lst.append(relative_percentage_error)
                    ccs3d.std_dev_list.append(std_dev)

                    # Evaluate the model for mean squared error
                    mse = mean_squared_error(predictions, ccs3d.test_exp_ccs[:truncate])
                    print(f"Mean Squared Error (MSE)  : {mse:.4f}")

                    mae = mean_absolute_error(ccs3d.test_exp_ccs[:truncate], predictions[:truncate])
                    print(
                        f"Mean Absolute Error (MAE) : {mae:.4f}",
                    )

                    mape = mean_absolute_percentage_error(
                        ccs3d.test_exp_ccs[:truncate], predictions[:truncate]
                    )
                    print(f"Mean Absolute Error (MAPE): {mape:.4f}")

                    pearson_corr, p_value = pearsonr(ccs3d.test_exp_ccs[:truncate], predictions)
                    print(f"Pearson Corr. Coefficient : {pearson_corr:.4f}")
                    print(
                        colored(
                            "-----------------------------------------------------------------------------",
                            "green",
                        )
                    )

        # ============= PyQT5 Graph section

        # Create a PyQtGraph window with three plots: Training Loss, Absolute Validation Error, and Mean Squared Error
        self.win = pg.GraphicsWindow(title="Training Progress")
        self.plot_loss = self.win.addPlot(title="Training Loss")
        self.plot_abs_val_error = self.win.addPlot(title="Absolute Validation Error")
        self.plot_mse = self.win.addPlot(title="Mean Squared Error")
        self.plot_rspe = self.win.addPlot(title="Average % RSE")

        # Create curves for training loss, absolute validation error, and mean squared error plots
        self.curve_loss = self.plot_loss.plot(pen="r", name="Training Loss")
        self.curve_abs_val_error = self.plot_abs_val_error.plot(
            pen="b", name="Absolute Validation Error"
        )
        self.curve_mse = self.plot_mse.plot(pen="g", name="Mean Squared Error")
        self.curve_rpe = self.plot_rspe.plot(pen="g", name="Average % RSE")

        # Set axis labels and title
        self.plot_loss.setLabel("left", "Loss")
        self.plot_loss.setLabel("bottom", "Epoch")
        self.plot_loss.setTitle("Training Loss")

        self.plot_abs_val_error.setLabel("left", "Absolute Val. Error")
        self.plot_abs_val_error.setLabel("bottom", "Epoch")
        self.plot_abs_val_error.setTitle("Absolute Validation Error")

        self.plot_mse.setLabel("left", "MSE")
        self.plot_mse.setLabel("bottom", "Epoch")
        self.plot_mse.setTitle("Mean Squared Error")

        self.plot_rspe.setLabel("left", "% Relative Std. Error")
        self.plot_rspe.setLabel("bottom", "Epoch(x5)")
        self.plot_rspe.setTitle("Perc. Relative Std. Error")

        self.win.resize(1280, 720)
        # self.win.show()

        class TrainingPlotter(tf.keras.callbacks.Callback):
            def __init__(self, accuracy, auroc, loss, epochs):
                """
                Initializes the plotter with the training data.

                Args:
                - accuracy (np.array): 1D array of accuracy values.
                - auroc (np.array): 1D array of AUROC values.
                - loss (np.array): 1D array of loss values.
                - epochs (np.array): 1D array of training epochs.
                """
                self.accuracy = accuracy
                self.auroc = auroc
                self.loss = loss
                self.epochs = epochs

            def plot_training_metrics(self):
                """
                Creates a PyQt window with two panels:
                - A 3D scatter plot of accuracy, AUROC, and loss
                - A 2D line plot of loss versus epoch
                """
                # Create the Qt Application and window
                app = QApplication([])

                # Create a window with two panels (3D plot + 2D plot)
                win = QMainWindow()
                central_widget = QWidget()
                win.setCentralWidget(central_widget)
                layout = QHBoxLayout()
                central_widget.setLayout(layout)

                # Create a Matplotlib figure for the 3D plot
                fig_3d = plt.figure(figsize=(10, 8))
                ax_3d = fig_3d.add_subplot(111, projection="3d")

                # Create the 3D scatter plot (accuracy, auroc, loss)
                ax_3d.scatter(self.accuracy, self.auroc, self.loss, c="r", marker="o")
                ax_3d.set_xlabel("Accuracy")
                ax_3d.set_ylabel("AUROC")
                ax_3d.set_zlabel("Loss")
                ax_3d.set_title("3D Plot of Accuracy, AUROC, and Loss")

                # Create the canvas for the 3D plot and add it to the layout
                canvas_3d = FigureCanvas(fig_3d)
                layout.addWidget(canvas_3d)

                # Create a Matplotlib figure for the 2D loss vs epoch plot
                fig_2d = plt.figure(figsize=(10, 4))
                ax_2d = fig_2d.add_subplot(111)

                # Plot the loss vs epoch line chart
                ax_2d.plot(self.epochs, self.loss, color="b", label="Loss")
                ax_2d.set_xlabel("Epoch")
                ax_2d.set_ylabel("Loss")
                ax_2d.set_title("Loss vs Epoch")
                ax_2d.legend()

                # Create the canvas for the 2D plot and add it to the layout
                canvas_2d = FigureCanvas(fig_2d)
                layout.addWidget(canvas_2d)

                # Show the window
                win.show()

        # Callback function to update the PyQtGraph plots during training
        class GraphUpdateCallback(tf.keras.callbacks.Callback):
            def __init__(
                self,
                plot_loss,
                plot_abs_val_error,
                plot_mse,
                plot_rsep,
                curve_loss,
                curve_abs_val_error,
                curve_mse,
                curve_rpe,
            ):
                ccs3d.plot_loss = plot_loss
                ccs3d.plot_abs_val_error = plot_abs_val_error
                ccs3d.plot_mse = plot_mse
                ccs3d.plot_rsep = plot_rsep

                ccs3d.curve_loss = curve_loss
                ccs3d.curve_abs_val_error = curve_abs_val_error
                ccs3d.curve_mse = curve_mse
                ccs3d.curve_rpe = curve_rpe

            def make_plot(self):
                if len(ccs3d.val_mae_lst) < 2:
                    return
                # =================== Mean Absolute Error
                ccs3d.plot_val_mae.removeItem(ccs3d.gfx1)
                ccs3d.datos = pg.ScatterPlotItem()
                ccs3d.gfx1 = ccs3d.plot_val_mae.addPlot(title="Absolute MVE")
                ccs3d.gfx1.setLabel("left", "Abs. mean validation error")
                ccs3d.gfx1.setLabel("bottom", "epoch(s)")
                ccs3d.datos = ccs3d.gfx1.plot(pen="y")
                ccs3d.datos.setData(ccs3d.val_mae_lst)
                ccs3d.gfx1.enableAutoRange(x=True, y=True)
                # ==================== Mean Absolute Percentage Error
                ccs3d.plot_val_mape.removeItem(ccs3d.gfx2)
                ccs3d.datos = pg.ScatterPlotItem()
                ccs3d.gfx2 = ccs3d.plot_val_mape.addPlot(
                    title="Validation MAPE"
                )  #   val_mean_absolute_percentage_error
                ccs3d.gfx2.setLabel("left", "Abs. mean validation % error")
                ccs3d.gfx2.setLabel("bottom", "epoch(s)")
                ccs3d.datos = ccs3d.gfx2.plot(pen="y")
                ccs3d.datos.setData(ccs3d.val_mape_lst)
                ccs3d.gfx2.enableAutoRange(x=True, y=True)
                # ===============================

            def on_epoch_end(self, epoch, logs=None):
                # IMPORTANT:
                # Do not append to history lists here. HistoryCallback already records
                # one value per epoch. Appending again here caused 250-epoch runs to
                # appear as 500 points in the GUI loss plot.

                # Update the curves using the histories recorded by HistoryCallback.
                ccs3d.curve_loss.setData(ccs3d.history_loss)
                ccs3d.curve_abs_val_error.setData(ccs3d.history_mean_val_abs_error)
                ccs3d.curve_mse.setData(ccs3d.history_mse)
                ccs3d.curve_rpe.setData(ccs3d.history_rpe)

                # Refresh the plots
                ccs3d.plot_loss.enableAutoRange("y")
                ccs3d.plot_abs_val_error.enableAutoRange("y")
                ccs3d.plot_mse.enableAutoRange("y")
                ccs3d.plot_rsep.enableAutoRange("y")

                # Process pending events to prevent freezing
                QtGui.QApplication.processEvents()
                # =============================
                ccs3d.update_progressBar(ccs3d.train_epoch, epoch + 1)
                # ============================
                self.make_plot()

        # Create the graph update callback
        self.graph_update_callback = GraphUpdateCallback(
            self.plot_loss,
            self.plot_abs_val_error,
            self.plot_mse,
            self.plot_rspe,
            self.curve_loss,
            self.curve_abs_val_error,
            self.curve_mse,
            self.curve_rpe,
        )

        # Create an instance of  3D TrainingPlotter and call the plotting method
        self.ccs3d_plotter = TrainingPlotter(
            self.plot_abs_val_error, self.plot_mse, self.plot_loss, self.train_epoch
        )

        # Create the history callback
        self.history_callback = HistoryCallback()

        self.lr_decay = LearningRateScheduler(
            self.lr_scheduler
        )  # Create the LearningRateScheduler callback

        if self.test_percent < 0.05:
            print(
                colored(
                    "WARNING! RPE evaluation on validation set WILL NOT be conducted during training session due to validation | test set < 5% of trainin set",
                    "red",
                )
            )

        # ====================================================================================================== main train part
        if (self.train_epoch - self.last_epoch) > 0:
            # Fit the model with checkpoint callback, history callback, and store the training history
            training_fit_start_perf = time.perf_counter()
            self.history = self.regression_model.fit(
                self.train_data,
                self.train_exp_ccs,
                batch_size=self.batch_size,
                epochs=(self.train_epoch - self.last_epoch),
                validation_data=(self.val_data, self.val_exp_ccs),
                callbacks=[
                    self.model_checkpoint_callback,
                    self.history_callback,
                    self.graph_update_callback,
                    self.lr_decay,
                    self.ccs3d_plotter,
                ],
                verbose=0,
            )
            training_fit_elapsed_min = (time.perf_counter() - training_fit_start_perf) / 60.0
        else:
            print(
                colored(
                    f"#ERROR! Model already traiend to target epoch {self.last_epoch}. Increase the target epoch to train further",
                    "red",
                )
            )
            return
        # ========================================================================================================

        os.makedirs(self.evaluations_dir, exist_ok=True)

        print(colored("\nStoring trained model backup at training end:", "yellow"))

        # ======================================= Store training curves and RPE history
        x_rspe = [x for x in self.history_rpe]
        y_epoch = list(range(len(x_rspe)))

        # RPE is evaluated every self.eval_train_freq epochs, so the first point is
        # eval_train_freq, not epoch 0.
        df_rspe = pd.DataFrame(
            {
                "source_data": self.csv_file_path,
                "dataset_id": self.dataset_id,
                "random_seed": self.random_seed,
                "batch_size": self.batch_size,
                "target_epoch": self.train_epoch,
                "epoch": [(x + 1) * self.eval_train_freq for x in y_epoch],
                "rpe": x_rspe,
                "rpe_std_dev": self.std_dev_list[: len(x_rspe)],
            }
        )

        print(df_rspe.head())
        # ======================================

        rpe_csv_path = os.path.join(
            self.evaluations_dir, f"rpe_data_bs{self.batch_size}_epoch{self.train_epoch}.csv"
        )
        df_rspe.to_csv(rpe_csv_path, index=False)

        # Save the data behind the GUI loss graph to Excel.
        # One row = one recorded epoch. After the callback fix, a 250-epoch run
        # should produce exactly 250 rows unless training was intentionally resumed
        # from an existing model/configuration.
        history_len = max(
            len(self.history_loss),
            len(self.history_val_loss),
            len(self.history_mse),
            len(self.history_val_mse),
            len(self.history_mean_val_abs_error),
            len(self.val_mape_lst),
        )

        def _pad_history(values, target_len):
            values = list(values)
            return values + [np.nan] * (target_len - len(values))

        df_training_history = pd.DataFrame(
            {
                "source_data": [self.csv_file_path] * history_len,
                "dataset_id": [self.dataset_id] * history_len,
                "random_seed": [self.random_seed] * history_len,
                "batch_size": [self.batch_size] * history_len,
                "target_epoch": [self.train_epoch] * history_len,
                "epoch": list(range(1, history_len + 1)),
                "train_loss": _pad_history(self.history_loss, history_len),
                "validation_loss": _pad_history(self.history_val_loss, history_len),
                "train_mse": _pad_history(self.history_mse, history_len),
                "validation_mse": _pad_history(self.history_val_mse, history_len),
                "validation_mae": _pad_history(self.history_mean_val_abs_error, history_len),
                "validation_mape": _pad_history(self.val_mape_lst, history_len),
            }
        )

        df_training_summary = pd.DataFrame(
            {
                "parameter": [
                    "source_data",
                    "dataset_id",
                    "random_seed",
                    "batch_size",
                    "target_epoch",
                    "recorded_epoch_rows",
                    "train_ratio",
                    "internal_validation_ratio",
                    "internal_test_ratio",
                    "rpe_evaluation_frequency",
                    "note",
                ],
                "value": [
                    self.csv_file_path,
                    self.dataset_id,
                    self.random_seed,
                    self.batch_size,
                    self.train_epoch,
                    history_len,
                    self.train_percent,
                    self.val_percent,
                    self.test_percent,
                    self.eval_train_freq,
                    "One row in epoch_history equals one epoch recorded by HistoryCallback. GraphUpdateCallback no longer appends duplicate history values.",
                ],
            }
        )

        loss_excel_path = os.path.join(
            self.evaluations_dir,
            f"{self.dataset_id}_rseed{self.random_seed}_loss_history_bs{self.batch_size}_epoch{self.train_epoch}_{self.timestamp}.xlsx",
        )

        try:
            with pd.ExcelWriter(loss_excel_path, engine="openpyxl") as writer:
                df_training_summary.to_excel(writer, sheet_name="summary", index=False)
                df_training_history.to_excel(writer, sheet_name="epoch_history", index=False)
                df_rspe.to_excel(writer, sheet_name="rpe_history", index=False)
            print(colored(f"Training loss history Excel stored at: {loss_excel_path}", "green"))
        except Exception as e:
            print(colored(f"WARNING! Could not save training loss history Excel file: {e}", "red"))

        print(colored(f"Training RPE data stored at: {rpe_csv_path}", "green"))
        print("Training data stored----")

        # Store at the final model at end of training regardless of the optimzied weights

        # ============ Update regression mdoel filename
        self.reg_model_fname = os.path.join(
            self.dataset_id
            + f"_{self.img_dim}x{self.img_dim}_rseed{self.random_seed}_model_ckpt.h5"
        )
        # ==============

        self.regression_model.save_weights(os.path.join(self.store_weights, self.reg_model_fname))
        self.save_config(
            cfg_fpath=os.path.join(
                self.store_weights,
                os.path.splitext(os.path.basename(self.reg_model_fname))[0] + ".cfg",
            )
        )
        print(
            colored(
                f"#Model & configuration stored at end of training to : {self.store_weights}",
                "yellow",
            )
        )
        # ===============

        # Store at the Evaluation folder
        # ===============
        self.regression_model.save_weights(os.path.join(self.evaluations_dir, self.reg_model_fname))
        self.save_config(
            cfg_fpath=os.path.join(
                self.evaluations_dir,
                os.path.splitext(os.path.basename(self.reg_model_fname))[0] + ".cfg",
            )
        )

        # ============== Store pyqt loss graph
        self.save_win_image(
            os.path.join(
                self.evaluations_dir,
                f"{self.dataset_id}_rseed{self.random_seed}_trainer_losses.png",
            )
        )
        # ===============

        self.train_time = timed_elapsed_in_min(
            start_time=self.train_start_time
        )  # colpute train time

        training_total_elapsed_min = (time.perf_counter() - training_wall_start_perf) / 60.0
        training_summary_rows = [
            {
                "stage": "training",
                "source_data": self.csv_file_path,
                "dataset_id": self.dataset_id,
                "projection_dir": self.dataset_path,
                "model_output_dir": self.evaluations_dir,
                "random_seed": self.random_seed,
                "batch_size": self.batch_size,
                "target_epoch": self.train_epoch,
                "image_dimension": f"{self.img_dim}x{self.img_dim}",
                "num_train_samples": len(self.train_data),
                "num_internal_validation_samples": len(self.val_data),
                "num_internal_test_samples": len(self.test_data),
                "data_loading_splitting_and_model_compile_elapsed_min": getattr(
                    self, "training_data_processing_elapsed_min", np.nan
                ),
                "fit_elapsed_min": training_fit_elapsed_min,
                "total_training_function_elapsed_min_before_posttrain_inference": training_total_elapsed_min,
                "train_time_original_min": self.train_time,
                "note": "This file is written at the end of model training, before optional post-training inference is run.",
            }
        ]
        self.write_runtime_timing_excel(
            stage_name="training",
            summary_rows=training_summary_rows,
            detail_rows=[],
            output_dir=self.evaluations_dir,
            filename_prefix=f"{self.dataset_id}_rseed{self.random_seed}",
        )

        print(colored(f"\nTraining completed in {self.train_time} minutes:", "green"))
        print(
            colored(
                f"\n\n-----------------------------------------------------------------------------------------------",
                "white",
            )
        )

        # ==================

        if (
            self.test_percent > 0.05
        ):  # only do inference for post_train_evalulation if there is validation datsset
            self.inference_3dccs(
                post_train_evalulation=True
            )  # sliently use infererence on validation data after trainingh
        else:
            print(
                colored(
                    "#[WARNING] Skipping post train evalulation due to very low (<5%) or No validation samples ",
                    "red",
                )
            )
        # ==================

    # ============================ INFERENCE=============================================

    def inference_3dccs(self, post_train_evalulation=True, kfold=""):
        inference_wall_start_perf = time.perf_counter()
        inference_data_prep_elapsed_min = 0.0
        inference_model_load_elapsed_min = 0.0
        inference_evaluate_elapsed_min = 0.0
        inference_prediction_elapsed_min = 0.0
        inference_output_elapsed_min = 0.0

        self.post_train_evalulation = post_train_evalulation
        print("Running in post train evaulations :", self.post_train_evalulation)

        if self.post_train_evalulation == False:
            if (
                self.qm.question(
                    self,
                    "3CDCCS",
                    f"Run inference for CCS prediction?\nNOTE: Ensure the number of rotation and pixel resoultion to match training | inference dataset",
                    self.qm.Yes | self.qm.No,
                )
                == self.qm.Yes
            ):
                pass
            else:
                return
        self.run_mode = "Running sample CCS prediction"
        # ========================================================================================================
        if (
            post_train_evalulation == False
        ):  # Manually use traiend model for inference case, here test set = entire data
            print(colored(f"#Preforming manual inference..", "yellow"))
            self.update_gui_vars()
            # for manual inference set the csv and dataset path absed on the user inout files
            self.dataset_path = self.inference_projection_path.toPlainText()
            self.csv_file_path = self.smile_msdata_filepath.toPlainText()
            self.reg_model_fname = (
                self.trained_model.toPlainText()
            )  # only if Post rain evaulatin is false
            self.std_dev_list = [0, 0, 0, 0]  # set the sdtev lsi to NULL for inference
            self.img_dim = int(
                self.inf_img_dim.currentText()
            )  # use the input pixel resoultion by the user
            inference_data_prep_start_perf = time.perf_counter()
            self.process_data_for_inference()  # to get the self.test_data
            inference_data_prep_elapsed_min = (
                time.perf_counter() - inference_data_prep_start_perf
            ) / 60.0

            self.train_data = (
                self.test_data
            )  # let train and test data be same for inference for model shape for self.create_3dcnn_regression_model()
            self.num_train_samples = (
                self.num_samples
            )  # let train samples be total samples = val samples
            sample_index = 0  # we use whole dataset as inference  in inference only mode

            if self.exp_ccs_is_missing:
                print(colored("\n-----------------------------------\n", "white"))
                print(
                    colored(
                        "\nWARNING! All or some of the experimental CCS values are missing. Thus, the validation charts/RMPE graphs may not be relevant",
                        "red",
                    )
                )
                print(colored("\n-----------------------------------\n", "white"))

        elif post_train_evalulation == True:
            sample_index = (
                self.num_train_samples + self.num_val_samples
            )  # becuase the starting pouint ot test sample is (self.num_train_samples + self.num_val_samples)

        # self.reg_model_fname =  self.trained_model.toPlainText()                           # Load model if it exists, load weights

        inference_model_load_start_perf = time.perf_counter()
        if os.path.isfile(self.reg_model_fname):
            print(colored(f"#Loading trained model : {self.reg_model_fname}", "yellow"))

            self.regression_model = self.create_3dcnn_regression_model()
            if self.loss_func == "Huber":
                model_loss_func = Huber(delta=1.0)
            if self.loss_func == "MSE":
                model_loss_func = "mse"
            # Compile the models
            self.regression_model.compile(
                optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),  # Import Huber Loss
                loss=model_loss_func,
                metrics=[
                    "mean_squared_error",  # loss='mean_squared_error'
                    "mean_absolute_error",
                    "mean_absolute_percentage_error",
                ],
            )

            self.regression_model.load_weights(self.reg_model_fname)
            self.model_config = (
                os.path.splitext(self.reg_model_fname)[0] + ".cfg"
            )  # same name with .cfg as extension during inference

            # make Evaluation directory
            os.makedirs(self.evaluations_dir, exist_ok=True)

            if os.path.isfile(self.model_config):
                print(colored("#Model history found. Loading model train history..."))
                self.load_config(self.model_config)
                # =========================================================================================================

                # Summarized losses by showing graphs
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                axes[0].set_title(f"{self.dataset_id} | loss history")
                axes[0].set_xlabel("epoch")
                axes[0].set_ylabel("log(loss)")
                axes[0].set_yscale("log")
                (
                    axes[0].plot(self.history_loss, label="train loss")
                    if post_train_evalulation == False
                    else axes[0].plot(self.history.history["loss"], label="Train Loss")
                )
                (
                    axes[0].plot(self.history_val_loss, label="validation loss")
                    if post_train_evalulation == False
                    else axes[0].plot(self.history.history["val_loss"], label="Validation Loss")
                )
                axes[0].legend()

                axes[1].set_title(f"{self.dataset_id} {kfold}| mean squared error (mse) history")
                axes[1].set_xlabel("epoch")
                axes[1].set_ylabel("mean squared error")
                axes[1].set_yscale("log")
                (
                    axes[1].plot([np.log(mse) for mse in self.history_mse], label="train mse")
                    if post_train_evalulation == False
                    else axes[1].plot(
                        [np.log(mse) for mse in self.history.history["mean_squared_error"]],
                        label="train mse",
                    )
                )
                (
                    axes[1].plot(
                        [np.log(mse) for mse in self.history_val_mse], label="validation mse"
                    )
                    if post_train_evalulation == False
                    else axes[1].plot(
                        self.history.history["val_mean_squared_error"], label="validation mse"
                    )
                )
                axes[1].legend()

                axes[2].set_title(
                    f"{self.dataset_id} {kfold}| absolute validation error (ave) history"
                )
                axes[2].set_xlabel("epoch")
                axes[2].set_ylabel("absolute error")
                axes[2].plot(
                    self.history_mean_val_abs_error, label="absolute validation error", color="gray"
                )
                axes[2].legend()

                plt.tight_layout()
                plt.draw()
                plt.pause(0.001)
                plt.savefig(
                    os.path.join(
                        self.evaluations_dir,
                        f"{self.dataset_id}{kfold}_summary_losses_{self.timestamp}.png",
                    )
                )

        inference_model_load_elapsed_min = (
            time.perf_counter() - inference_model_load_start_perf
        ) / 60.0

        # ====================================================== MODEL EVALUATION =======================================

        # Evaluate the model on the test set
        inference_evaluate_start_perf = time.perf_counter()
        mse = self.regression_model.evaluate(self.test_data, self.test_exp_ccs, batch_size=1)
        inference_evaluate_elapsed_min = (
            time.perf_counter() - inference_evaluate_start_perf
        ) / 60.0
        print("Mean Squared Error (MSE) on Test Set:", mse)

        # Perform inference on the test set
        # self.predictions = self.regression_model.predict(self.test_data)

        inference_prediction_start_perf = time.perf_counter()
        truncate = 1000
        test_data = self.test_data[:truncate]  # Assuming ccs3d.test_data is a NumPy array
        num_slices = min(
            20, len(test_data)
        )  # Set Number of slices & Determine the size of each slice
        slice_size = len(test_data) // num_slices  # size of slice as integer
        predictions = []  # Initialize an empty list to collect predictions
        for i in range(num_slices):  # Perform predictions for each slice
            print(colored(f"Inference for slice# ", "green"), colored(f"{i}/{num_slices}", "blue"))
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size if i < num_slices - 1 else len(test_data)
            slice_predictions = ccs3d.regression_model.predict(
                test_data[start_idx:end_idx], verbose=1
            ).flatten()
            predictions.append(slice_predictions)

        self.predictions = np.concatenate(predictions)  # Combine all slices into a single array
        inference_prediction_elapsed_min = (
            time.perf_counter() - inference_prediction_start_perf
        ) / 60.0

        relative_percentage_error, std_dev, skipped_count, skipped_percentage = (
            percentage_std_error(self.test_exp_ccs, self.predictions)
        )  # mean RPE coems from here

        print(colored("\n-------[ Basic evaluation metrics | scores ]-------", "blue"))
        print(
            colored("Mean Relative % error (MRPE) :", "green"),
            colored(f"{relative_percentage_error:.4f} %", "white"),
        )  # relative_percentage_error here means mean realtive eprtcentage error per experiment

        # Calculate mean absolute error (MAE) on the test set
        mae = mean_absolute_error(self.test_exp_ccs, self.predictions)
        print(colored("Mean Absolute Error (MAE)    :", "green"), colored(mae, "white"))

        mape = mean_absolute_percentage_error(self.test_exp_ccs, self.predictions)
        print(colored("Mean Absolute Error (MAPE)   :", "green"), colored(mape, "white"))

        # Calculate Pearson correlation coefficient
        self.pearson_corr, p_value = pearsonr(self.test_exp_ccs, self.predictions.flatten())
        print(
            colored(f"{kfold}Pearson's Corr. Coeff.       :", "green"),
            colored(self.pearson_corr, "white"),
        )

        # Calculate the correlation matrix
        self.correlation_matrix = np.corrcoef(self.test_exp_ccs, self.predictions.flatten())

        # ====================================================================================== Display resulst in Inference table in GUI widget
        each_row = []

        for index, each_mol in enumerate(self.predictions):
            each_row.append(
                [
                    self.mol_name_list[index + sample_index],
                    self.mol_smiles_list[index + sample_index],
                    self.found_sample_names[index + sample_index],
                    str(self.exp_mz_ratio[index + sample_index]),
                    str(self.predictions[index]),
                    str(self.exp_ccs[index + sample_index]),
                ]
            )
        self.add_rows_to_table(each_row)

        # =================================================================================================================

        inference_output_start_perf = time.perf_counter()

        # Build scatter plot
        plt.figure(figsize=(12, 8))
        plt.scatter(
            self.test_exp_ccs, self.predictions.flatten(), color="black", label="Data Points"
        )
        plt.xlabel("True Experimental CCS values")
        plt.ylabel("Predicted CCS values")
        plt.title(f"{self.dataset_id} {kfold}| Exp. CCS values vs. Pred. CCS values ")
        plt.grid(True)

        # Perform linear regression and print coefficients
        slope, intercept, r_value, p_value, std_err = linregress(
            self.test_exp_ccs, self.predictions.flatten()
        )

        print("\n")
        print(
            colored(f"Linear regression metrics for :", "blue"), colored(self.dataset_id, "white")
        )
        print(colored("___________________________________________________", "green"))
        print("Slope                         :", slope)
        print("Intercept                     :", intercept)
        print("R-squared                     :", r_value**2)
        print("P-value                       :", p_value)
        print("Standard Error                :", std_err)
        print(colored("___________________________________________________", "green"))

        # Print linear regression coefficients
        plt.text(
            0.98,
            0.98,
            f"Pearson Corr.: {self.pearson_corr:.2f}\nSlope: {slope:.2f}\nIntercept: {intercept:.2f}\nR-squared: {r_value ** 2:.2f}\nStd Error: {std_err:.2f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.evaluations_dir, f"{self.dataset_id}_scatter_plot_{self.timestamp}.png"
            )
        )  # Save scatter plot

        # ======================================================================================================================================= Prediction Values CSVs
        print(colored("\n#Writing predicted CCS for validation set..", "blue"))
        result_file = os.path.join(
            self.evaluations_dir, f"{self.dataset_id}{kfold}_predictions_{self.timestamp}.csv"
        )
        with open(
            result_file, "w", encoding="latin-1"
        ) as f:  # encoding must be latin-1 or else there will be error

            if len(self.mol_name_list) > 0:
                # Write header header for new CSV file with molecule
                f.write(
                    f"index,ID,SMILE,name,predicted_CCS,experimental_CCS,relative_percentage_error"
                )
                for index, (true_ccs, predicted_ccs) in enumerate(
                    zip(self.test_exp_ccs, self.predictions.flatten())
                ):
                    mol_name = str(self.mol_name_list[index + sample_index]).replace(",", "-")
                    f.write(
                        f"\n{index},{self.found_sample_names[index + sample_index]}, {self.mol_smiles_list[index + sample_index]}, {mol_name},{predicted_ccs},{true_ccs}, {100 * abs(true_ccs - predicted_ccs)/true_ccs}"
                    )
            else:
                # Write header header for new CSV file without moelculae name
                f.write(f"index,ID,SMILE,predicted_CCS,experimental_CCS,relative_percentage_error")
                for index, (true_ccs, predicted_ccs) in enumerate(
                    zip(self.test_exp_ccs, self.predictions.flatten())
                ):
                    f.write(
                        f"\n{index},{self.found_sample_names[index + sample_index]}, {self.mol_smiles_list[index + sample_index]}, {predicted_ccs},{true_ccs}, {100 * abs(true_ccs - predicted_ccs)/true_ccs}"
                    )

        if (
            post_train_evalulation == True
        ):  # =====================SAVE Threshold cutoff source samples only for Post train evaulation
            filtered_output_fname = os.path.join(
                self.evaluations_dir,
                os.path.splitext(os.path.basename(self.SMILE_src_filepath))[0] + "_filtered.csv",
            )
            filtered_trainer_fname = os.path.join(
                self.evaluations_dir,
                os.path.splitext(os.path.basename(self.SMILE_src_filepath))[0]
                + "_trainer_filtered.csv",
            )
            # filterd_source_file(source_file = self.SMILE_src_filepath, result_file = result_file , output_file =filtered_output_fname , output_trainer = filtered_trainer_fname, error_threshold=3.0)
            print(f"\nDEGUG: {self.SMILE_src_filepath}\n\n")
            filterd_source_file(
                self.SMILE_src_filepath,
                result_file,
                filtered_output_fname,
                filtered_trainer_fname,
                3.0,
            )
            print(f"\nDEGUG: {self.SMILE_src_filepath}\n\n")

        # ====================================================================================BAR CHART
        # Save BAR CHART with real validation sample labels
        plt.figure(figsize=(20, 12))

        truncate = max(
            1, self.inf_barchart_truncate_slider.value()
        )  # for proper visbility reduce sample size in output
        ind = np.arange(len(self.test_exp_ccs[:truncate]))
        width = 0.35

        plt.bar(ind, self.test_exp_ccs[:truncate], width, label="Exp. CCS", color="black")
        plt.bar(
            ind + width,
            self.predictions.flatten()[:truncate],
            width,
            label="Predicted CCS",
            color="lightgray",
        )
        plt.xlabel("Validation samples", fontsize=18)
        plt.ylabel("CCS Value", fontsize=18)
        plt.title(f"{self.dataset_id} {kfold}| Exp. CCS values vs. Predicted CCS values")
        plt.xticks(ind + width / 2, range(len(self.test_exp_ccs[:truncate])))

        # Add labels for each sample (folder name)
        self.sample_names = [
            folder_name.capitalize()
            for folder_name in self.found_sample_names[sample_index:][:truncate]
        ]
        plt.xticks(
            ind + width / 2,
            self.sample_names[: len(ind)][:truncate],
            rotation=90,
            ha="right",
            fontsize=8,
        )  # Adjust font size

        # Add labels for each sample (folder name) and values on top of the bars
        for i, (true_ccs, predicted_ccs) in enumerate(
            zip(self.test_exp_ccs[:truncate], self.predictions.flatten()[:truncate])
        ):
            plt.text(
                i,
                max(true_ccs, predicted_ccs) + 0.05,
                f"{true_ccs:.2f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8,
            )
            plt.text(
                i + width,
                max(true_ccs, predicted_ccs) + 0.075,
                f"{predicted_ccs:.2f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8,
            )

        # Print linear regression coefficients
        plt.text(
            0.98,
            0.98,
            f"-----Linear Regression Coefficients-----\n#Slope: {slope:.2f}\n#Intercept: {intercept:.2f}\n#R-squared: {r_value ** 2:.2f}\n#Std Error: {std_err:.2f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.evaluations_dir,
                f"{self.dataset_id}{kfold}_barchart_with_sample_names{self.timestamp}.png",
            )
        )
        # ========================================================================================================================================

        # Save heatmap
        import seaborn as sns

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            self.correlation_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            xticklabels=["True CCS", "Predicted CCS"],
            yticklabels=["True CCS", "Predicted CCS"],
        )
        plt.title(f"{self.dataset_id}{kfold}_Pearson Corr. Coefficient Heatmap")

        # Print Pearson correlation coefficient
        plt.text(
            0.98,
            0.98,
            f"Pearson Correlation: {self.pearson_corr:.2f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.5),
        )
        plt.savefig(
            os.path.join(
                self.evaluations_dir, f"{self.dataset_id}{kfold}_heatmap_{self.timestamp}.png"
            )
        )

        def find_matching_rotation(
            fname,
        ):  # Returns the first pattern found in fname, or None if no match exists
            patterns = ["5rot", "3rot", "1rot", "10rot", "2rot"]
            return next((p for p in patterns if p in fname), None)

        def find_matching_resoultion(
            fname,
        ):  # Returns the first pattern found in fname, or None if no match exists
            patterns = [
                "16x16",
                "18x18",
                "32x32",
                "64x64",
                "128x128",
                "192x192",
                "220x220",
                "256x256",
                "512x512",
            ]
            return next((p for p in patterns if p in fname), None)

        # ===================================================================
        # Prepare data dictionary for json file during inference (self.post_train_evalulation =False)
        data_inf = {
            "Results of inference": {
                "Inf. source datafile": self.smile_msdata_filepath.toPlainText(),
                "Inf. 2d projection": self.dataset_path,
                "Inf. model": self.trained_model.toPlainText(),
                "Inf. model resolution": f"{find_matching_resoultion(self.trained_model.toPlainText())} pixels",  # (just for now) f"{find_matching_resoultion(self.trained_model.toPlainText())} pixels",
                "Inf. model rotation": f"{find_matching_rotation(self.trained_model.toPlainText())}.",
                "Inf. pixel dims.": f"{self.inf_img_dim.currentText()}x{self.inf_img_dim.currentText()} pixels",
                "Adduct type (user set)": self.inf_adduct_type,
                "Pearson Correlation": f"{self.pearson_corr:.2f}",
                "Linear Regression Coefficients": {
                    "Slope": round(slope, 2),
                    "Intercept": round(intercept, 2),
                    "R-squared": round(r_value**2, 2),
                    "Std Error": round(std_err, 2),
                },
                "Mean Relative Percentage Error": {
                    "Mean Relative Percentage Error (MRPE)": round(relative_percentage_error, 4),
                    "StdDev of MRPE": round(std_dev, 4),
                },
                "Mean Absolute Deviation": round(np.mean(self.std_dev_list), 4),
            }
        }
        # =====================================================================

        # ===================================================================
        # Prepare data dictionary for json file for post_train_evalulation = True
        data_train = {
            "Results of post-train Evaluation": {
                "Source datafile": self.csv_file_path,
                "2d projection": self.dataset_path,
                "Model": self.reg_model_fname,
                "Random seed": self.random_seed,
                "Model resoultion": f"{self.img_dim}x{self.img_dim} pixels",
                "Model rotation": f"{find_matching_rotation(self.reg_model_fname)} rotations",
                "Pearson Correlation": f"{self.pearson_corr:.2f}",
                "Linear Regression Coefficients": {
                    "Slope": round(slope, 2),
                    "Intercept": round(intercept, 2),
                    "R-squared": round(r_value**2, 2),
                    "Std Error": round(std_err, 2),
                },
                "Mean Relative Percentage Error": {
                    "Mean Relative Percentage Error (MRPE)": round(relative_percentage_error, 4),
                    "StdDev of MRPE": round(std_dev, 4),
                },
                "Mean Absolute Deviation": round(np.mean(ccs3d.std_dev_list), 4),
            }
        }
        # =====================================================================
        # Optionally add train time if it exists
        try:
            data_train["Results of inference"]["Total train time (min)"] = self.train_time
        except:
            pass

        # Save as JSON file
        json_path = os.path.join(
            self.evaluations_dir,
            f"{self.dataset_id}{kfold}_linear_regression_{self.timestamp}.json",
        )

        with open(json_path, "w") as f:
            (
                json.dump(data_train, f, indent=4)
                if self.post_train_evalulation == True
                else json.dump(data_inf, f, indent=4)
            )

        # ==================================================================

        from scipy.stats import probplot

        # Calculate the residuals
        self.residuals = self.test_exp_ccs - self.predictions.flatten()

        # Create a Q-Q plot
        plt.figure(figsize=(8, 6))
        probplot(self.residuals, plot=plt)
        plt.title(f"{self.dataset_id}{kfold}_Quantile-Quantile Plot of Residuals")
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Sample Quantiles")
        plt.tight_layout()
        # Calculate Pearson correlation coefficient
        self.pearson_corr, _ = pearsonr(self.test_exp_ccs, self.predictions.flatten())
        # Annotate the plot with Pearson correlation coefficient
        plt.text(
            0.02,
            0.98,
            f"Pearson Correlation: {self.pearson_corr:.2f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(facecolor="white", alpha=0.5),
        )
        plt.show()
        plt.savefig(
            os.path.join(self.evaluations_dir, f"{self.dataset_id}{kfold}_{self.timestamp}.png")
        )

        # Calculate the residuals
        self.residuals = self.test_exp_ccs - self.predictions.flatten()

        # Create a density plot
        plt.figure(figsize=(8, 6))
        sns.histplot(self.residuals, kde=True)
        plt.title(f"{self.dataset_id} {kfold}| Density Plot of Residuals")
        plt.xlabel("Residuals")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.show()
        plt.savefig(
            os.path.join(
                self.evaluations_dir, f"{self.dataset_id}{kfold}_Desnityplot_{self.timestamp}.png"
            )
        )

        # Keep the plot window open
        (
            plt.ioff() if post_train_evalulation == False else None
        )  # do not close auotmatically for inference
        plt.show()
        plt.close("all")
        plt.ioff()
        # ==============
        inference_output_elapsed_min = (time.perf_counter() - inference_output_start_perf) / 60.0
        inference_total_elapsed_min = (time.perf_counter() - inference_wall_start_perf) / 60.0
        inference_summary_rows = [
            {
                "stage": "inference",
                "post_train_evalulation": self.post_train_evalulation,
                "dataset_id": self.dataset_id,
                "source_data": self.csv_file_path,
                "projection_dir": self.dataset_path,
                "model_file": self.reg_model_fname,
                "image_dimension": f"{self.img_dim}x{self.img_dim}",
                "num_inference_samples": len(self.test_data),
                "num_predicted_samples": len(self.predictions),
                "data_preparation_elapsed_min": inference_data_prep_elapsed_min,
                "model_load_and_config_elapsed_min": inference_model_load_elapsed_min,
                "model_evaluate_elapsed_min": inference_evaluate_elapsed_min,
                "prediction_elapsed_min": inference_prediction_elapsed_min,
                "plot_and_output_elapsed_min": inference_output_elapsed_min,
                "total_inference_function_elapsed_min": inference_total_elapsed_min,
                "mean_relative_percentage_error": relative_percentage_error,
                "std_relative_percentage_error": std_dev,
                "pearson_correlation": self.pearson_corr,
            }
        ]
        self.write_runtime_timing_excel(
            stage_name="inference",
            summary_rows=inference_summary_rows,
            detail_rows=[],
            output_dir=self.evaluations_dir,
            filename_prefix=f"{self.dataset_id}{kfold}",
        )

        self.clear_gpu_mem()

    # ============================ K_FOLD INFERENCE=============================================
    # ============================ K_FOLD INFERENCE=============================================
    # ============================ K_FOLD INFERENCE=============================================

    def inference_3dccs_kfold(
        self, post_train_evalulation=True, kfold=""
    ):  # only for inference busing traiend model

        if post_train_evalulation == True:  # do not conduct if not post-rpocessing
            pass
        else:
            return

        os.makedirs(self.evaluations_dir, exist_ok=True)  # make Evaluation directory

        if os.path.isfile(self.reg_model_fname):
            print(colored(f"#Loading configuration from : {self.reg_model_fname}", "yellow"))

            self.regression_model = self.create_3dcnn_regression_model()  # laod model archicture

            if self.loss_func == "Huber":
                model_loss_func = Huber(delta=1.0)
            if self.loss_func == "MSE":
                model_loss_func = "mse"
            # Compile the models

            self.regression_model.compile(
                optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate),  # Import Huber Loss
                loss=model_loss_func,
                metrics=[
                    "mean_squared_error",  # loss='mean_squared_error'
                    "mean_absolute_error",
                    "mean_absolute_percentage_error",
                ],
            )

            self.regression_model.load_weights(self.reg_model_fname)
            self.model_config = (
                os.path.splitext(self.reg_model_fname)[0] + ".cfg"
            )  # same name with .cfg as extension during inference

        # ====================================================== MODEL EVALUATION =======================================

        # Evaluate the model on the k fold validation set based on the [self.val_idx]
        mse = self.regression_model.evaluate(
            ccs3d.data[self.val_idx], ccs3d.exp_ccs[self.val_idx], batch_size=1
        )
        print("Mean Squared Error (MSE) on Test Set:", mse)

        # Perform inference on the test set. the inference is doen in slciece to avoid memory error.
        # self.predictions = self.regression_model.predict(self.test_data)

        truncate = len(self.val_idx)
        test_data = ccs3d.data[
            self.val_idx
        ]  # Assuming ccs3d.datais a NumPy array. we use here ccs3d.data becuase ofr k-fold artichecture
        num_slices = min(
            20, len(test_data)
        )  # Set Number of slices & Determine the size of each slice
        slice_size = len(test_data) // num_slices  # size of slice as integer
        predictions = []  # Initialize an empty list to collect predictions
        for i in range(num_slices):  # Perform predictions for each slice
            print(colored(f"Inference for slice# ", "green"), colored(f"{i}/{num_slices}", "blue"))
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size if i < num_slices - 1 else len(test_data)
            slice_predictions = ccs3d.regression_model.predict(
                test_data[start_idx:end_idx], verbose=1
            ).flatten()
            predictions.append(slice_predictions)

        self.predictions = np.concatenate(predictions)  # Combine all slices into a single array

        relative_percentage_error, std_dev, skipped_count, skipped_percentage = (
            percentage_std_error(ccs3d.exp_ccs[self.val_idx], self.predictions)
        )  # mean RPE coems from here
        print("----------------------------------------------")
        print(
            f"Percentage standard deviation: {relative_percentage_error:.4f},\nStandard Deviation: {std_dev:.4f}",
        )

        # Calculate mean absolute error (MAE) on the test set
        mae = mean_absolute_error(ccs3d.exp_ccs[self.val_idx], self.predictions)
        print("Mean Absolute Error (MAE) on Test Set:", mae)

        mape = mean_absolute_percentage_error(ccs3d.exp_ccs[self.val_idx], self.predictions)
        print("Mean Absolute Percentage Error (MAPE) on Test Set:", mape)

        # Calculate Pearson correlation coefficient
        self.pearson_corr, p_value = pearsonr(
            ccs3d.exp_ccs[self.val_idx], self.predictions.flatten()
        )
        print(f"{kfold}Pearson Correlation Coefficient: {self.pearson_corr}")

        # Calculate the correlation matrix
        self.correlation_matrix = np.corrcoef(
            ccs3d.exp_ccs[self.val_idx], self.predictions.flatten()
        )

        # ====================================================================================== Display resulst in Inference table in GUI widget
        each_row = []

        for index, each_mol in enumerate(self.predictions):
            each_row.append(
                [
                    self.mol_name_list[self.val_idx[index]],
                    self.mol_smiles_list[self.val_idx[index]],
                    self.found_sample_names[self.val_idx[index]],
                    str(self.exp_mz_ratio[self.val_idx[index]]),
                    str(self.predictions[index]),
                    str(self.exp_ccs[self.val_idx[index]]),
                ]
            )

        self.add_rows_to_table(each_row)

        # =================================================================================================================Save model for later use
        if True:  # turn on
            print(colored("\n#Saving model for future use:\nSaving model....:", "yellow"))
            self.regression_model.save_weights(
                os.path.join(self.evaluations_dir, self.dataset_id + f"{kfold}model_ckpt.h5")
            )  # (self.reg_model_fname)
            # self.save_config(cfg_fpath = os.path.join(self.evaluations_dir ,  self.dataset_id + f"_{kfold}_model_ckpt.cfg"))
            print(colored("\n#Model saved...", "yellow"))
            # self.save_config(self.model_config)  # saving general config

        # =================================================================================================================

        # Build scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(
            ccs3d.exp_ccs[self.val_idx],
            self.predictions.flatten(),
            color="black",
            label="Data Points",
        )
        plt.xlabel("True Experimental CCS values", fontsize=16)
        plt.ylabel("Predicted CCS values", fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.title(f"{self.dataset_id} {kfold}| Exp. CCS vs. Pred. CCS values")
        plt.grid(True)

        # Perform linear regression and print coefficients
        slope, intercept, r_value, p_value, std_err = linregress(
            ccs3d.exp_ccs[self.val_idx], self.predictions.flatten()
        )
        print(colored("___________________________________________________", "green"))
        print("Dataset ID                    :", self.dataset_id)
        print("Linear regression coefficients")
        print("Slope                         :", slope)
        print("Intercept                     :", intercept)
        print("R-squared                     :", r_value**2)
        print("P-value                       :", p_value)
        print("Standard Error                :", std_err)
        print(colored("___________________________________________________", "green"))

        # Print linear regression coefficients
        plt.text(
            0.96,
            0.92,
            f"Pearson Corr.: {self.pearson_corr:.2f}\nSlope: {slope:.2f}\nIntercept: {intercept:.2f}\nR-squared: {r_value ** 2:.2f}\nStd Error: {std_err:.2f}",
            transform=plt.gca().transAxes,
            fontsize=14,
            verticalalignment="bottom",
            horizontalalignment="left",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        plt.tight_layout()
        plt.legend()
        plt.savefig(
            os.path.join(
                self.evaluations_dir, f"{self.dataset_id}{kfold}_scatter_plot_{self.timestamp}.png"
            )
        )  # Save scatter plot

        # ======================================================================================================================================= Prediction Values CSVs
        print(colored("\n#Writing predicted CCS for validation set..", "blue"))
        result_file = os.path.join(
            self.evaluations_dir, f"{self.dataset_id}{kfold}_predictions_{self.timestamp}.csv"
        )
        with open(result_file, "w") as f:

            if len(self.mol_name_list) > 0:
                # Write header header for new CSV file with molecule
                f.write(
                    f"index,ID,SMILE,name,predicted_CCS,experimental_CCS,relative_percentage_error"
                )
                for index, (true_ccs, predicted_ccs) in enumerate(
                    zip(ccs3d.exp_ccs[self.val_idx], self.predictions.flatten())
                ):
                    mol_name = str(self.mol_name_list[self.val_idx[index]]).replace(",", "-")
                    f.write(
                        f"\n{index},{self.found_sample_names[self.val_idx[index]]}, {self.mol_smiles_list[self.val_idx[index]]}, {mol_name},{predicted_ccs},{true_ccs}, {100 * abs(true_ccs - predicted_ccs)/true_ccs}"
                    )
            else:
                # Write header header for new CSV file without moelculae name
                f.write(f"index,ID,SMILE,predicted_CCS,experimental_CCS,relative_percentage_error")
                for index, (true_ccs, predicted_ccs) in enumerate(
                    zip(ccs3d.exp_ccs[self.val_idx], self.predictions.flatten())
                ):
                    f.write(
                        f"\n{index},{self.found_sample_names[self.val_idx[index]]}, {self.mol_smiles_list[self.val_idx[index]]}, {predicted_ccs},{true_ccs}, {100 * abs(true_ccs - predicted_ccs)/true_ccs}"
                    )

        # =====================SAVE Threshold cutoff Source samples TODO
        filtered_output_fname = os.path.join(
            self.evaluations_dir,
            os.path.splitext(os.path.basename(self.SMILE_src_filepath))[0]
            + f"{kfold}_filtered.csv",
        )
        filtered_trainer_fname = os.path.join(
            self.evaluations_dir,
            os.path.splitext(os.path.basename(self.SMILE_src_filepath))[0]
            + f"{kfold}_trainer_filtered.csv",
        )
        filterd_source_file(
            source_file=self.SMILE_src_filepath,
            result_file=result_file,
            output_file=filtered_output_fname,
            output_trainer=filtered_trainer_fname,
            error_threshold=3.0,
        )

        # ====================================================================================BAR CHART
        # Save BAR CHART with real validation sample labels
        plt.figure(figsize=(20, 12))

        truncate = max(
            1, self.inf_barchart_truncate_slider.value()
        )  # for proper visbility reduce sample size in output
        ind = np.arange(len(ccs3d.exp_ccs[self.val_idx]))
        width = 0.35

        plt.bar(ind, ccs3d.exp_ccs[self.val_idx], width, label="Exp. CCS", color="black")
        plt.bar(
            ind + width, self.predictions.flatten(), width, label="Predicted CCS", color="lightgray"
        )
        plt.xlabel("Validation samples", fontsize=14)
        plt.ylabel("CCS Value", fontsize=14)
        plt.title(f"{self.dataset_id} {kfold}| Exp. CCS values vs. Predicted CCS values")
        plt.xticks(ind + width / 2, range(len(ccs3d.exp_ccs[self.val_idx])))

        # Add labels for each sample (folder name)
        self.sample_names = [folder_name.capitalize() for folder_name in self.found_sample_names]
        plt.xticks(
            ind + width / 2, self.sample_names[: len(ind)], rotation=90, ha="right", fontsize=8
        )  # Adjust font size

        # Add labels for each sample (folder name) and values on top of the bars
        for i, (true_ccs, predicted_ccs) in enumerate(
            zip(ccs3d.exp_ccs[self.val_idx], self.predictions.flatten())
        ):
            plt.text(
                i,
                max(true_ccs, predicted_ccs) + 0.05,
                f"{true_ccs:.2f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8,
            )
            plt.text(
                i + width,
                max(true_ccs, predicted_ccs) + 0.075,
                f"{predicted_ccs:.2f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8,
            )

        # Print linear regression coefficients
        plt.text(
            0.98,
            0.98,
            f"-----Linear Regression Coefficients-----\n#Slope: {slope:.2f}\n#Intercept: {intercept:.2f}\n#R-squared: {r_value ** 2:.2f}\n#Std Error: {std_err:.2f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                self.evaluations_dir,
                f"{self.dataset_id}{kfold}_barchart_with_sample_names{self.timestamp}.png",
            )
        )
        # ========================================================================================================================================

        # Save heatmap
        import seaborn as sns

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            self.correlation_matrix,
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            xticklabels=["True CCS", "Predicted CCS"],
            yticklabels=["True CCS", "Predicted CCS"],
        )
        plt.title(f"{self.dataset_id}{kfold}_Pearson Corr. Coefficient Heatmap")

        # Print Pearson correlation coefficient
        plt.text(
            0.98,
            0.98,
            f"Pearson Correlation: {self.pearson_corr:.2f}",
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(facecolor="white", alpha=0.5),
        )
        plt.savefig(
            os.path.join(
                self.evaluations_dir, f"{self.dataset_id}{kfold}_heatmap_{self.timestamp}.png"
            )
        )

        kfold_info = "N/A" if kfold == "" else kfold
        # Prepare data dictionary for json file
        data = {
            "Results of inference": {
                "Dataset ID": self.dataset_id,
                "Adduct type": self.inf_adduct_type,
                "K-folding": kfold_info,
                "Source datafile": self.csv_file_path,  # self.SMILE_src_filepath,
                "2D Projection data": self.dataset_path,
                "Random seed": self.random_seed,
                "Surface binarization": self.image_threshold,
                "Train epoch": self.train_epoch,
                "Loss function": self.loss_func,
                "Image dimension": f"{self.img_dim}x{self.img_dim} pixels",
                "Linear Regression Coefficients": {
                    "Slope": round(slope, 2),
                    "Intercept": round(intercept, 2),
                    "R-squared": round(r_value**2, 2),
                    "Std Error": round(std_err, 2),
                },
                "Mean Relative Percentage Error": {
                    "Mean": round(relative_percentage_error, 4),
                    "StdDev": round(std_dev, 4),
                },
                "Mean Absolute error": round(mae, 4),
            }
        }

        # Optionally add train time if it exists
        try:
            data["Results of inference"]["Total train time (min)"] = self.train_time
        except:
            pass

        # Save as JSON file
        json_path = os.path.join(
            self.evaluations_dir,
            f"{self.dataset_id}{kfold}_linear_regression_{self.timestamp}.json",
        )

        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)

        # ==================================================================

        from scipy.stats import probplot

        # Calculate the residuals
        self.residuals = self.exp_ccs[self.val_idx] - self.predictions.flatten()

        # Create a Q-Q plot
        plt.figure(figsize=(6, 6))
        probplot(self.residuals, plot=plt)
        plt.title(f"{self.dataset_id}{kfold}_Quantile-Quantile Plot of Residuals")
        plt.xlabel("Theoretical Quantiles")
        plt.ylabel("Sample Quantiles")
        plt.tight_layout()
        # Calculate Pearson correlation coefficient
        self.pearson_corr, _ = pearsonr(self.exp_ccs[self.val_idx], self.predictions.flatten())
        # Annotate the plot with Pearson correlation coefficient
        plt.text(
            0.015,
            0.98,
            f"Pearson Correlation: {self.pearson_corr:.2f}",
            transform=plt.gca().transAxes,
            fontsize=15,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(facecolor="white", alpha=0.5),
        )
        plt.show()
        plt.savefig(
            os.path.join(self.evaluations_dir, f"{self.dataset_id}{kfold}_{self.timestamp}.png")
        )

        # Create a density plot
        plt.figure(figsize=(8, 6))
        sns.histplot(self.residuals, kde=True)
        plt.title(f"{self.dataset_id} {kfold}| Density Plot of Residuals")
        plt.xlabel("Residuals")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.show()
        plt.savefig(
            os.path.join(
                self.evaluations_dir, f"{self.dataset_id}{kfold}_Desnityplot_{self.timestamp}.png"
            )
        )

        # Keep the plot window open
        (
            plt.ioff() if post_train_evalulation == False else None
        )  # do not close auotmatically for inference
        plt.show()
        plt.close("all")
        plt.ioff()
        # ==============
        self.clear_gpu_mem()

    def exit_3dccs(self):
        if self.qm.question(self, "CDCCS", f"Quit 3DCCS?", self.qm.Yes | self.qm.No) == self.qm.Yes:
            print("Exiting 3DCCCS.")
            try:
                self.save_config(
                    cfg_fpath=os.path.join(
                        self.evaluations_dir,
                        os.path.splitext(os.path.basename(self.reg_model_fname))[0] + ".cfg",
                    )
                )
                self.clear_gpu_mem()
                sys.exit(-1)
            except:
                sys.exit(-1)

            return
        return


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    qtStyle = ["Breeze", "Oxygen", "QtCurve", "Windows", "Fusion"]
    app.setStyle(qtStyle[4])  # [ 0: Breeze, 1: Oxygen, 2: QtCurve, 3: Windows, 4:Fusion ]'
    ccs3d = MyApp()
    ccs3d.show()
    warnings.filterwarnings(
        "ignore"
    )  # supress unnecessary/minor  warnings in console output  (version, depreciation stuffs like that)
    ccs3d.setWindowTitle(
        "Deep3DCCS: Geometry-Driven DL Approach for Lipid CCS Prediction from Multi-Angle Molecular Projections"
    )
    print(colored(f"Running on GUI interface styled with:  {qtStyle[4]}", "blue"))
    sys.exit(app.exec_())
