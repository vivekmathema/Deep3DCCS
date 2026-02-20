'''
uyility_modules.py
GUI utility modules and base classes for Deep3DCCS PyQt5 interface. Extends main 
application with specialized widgets, file dialogs, and user interaction handlers. 
Manages variable synchronization between GUI controls and backend processing parameters. 
Implements image preview, GPU configuration, and progress bar updating functionality. 
Provides reusable components for consistent user experience across application tabs.
'''
import os, sys
import time ,glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DETERMINISTIC_OPS'] = '1'  # Set the environment variable for deterministic TensorFlow operations
import csv
from csv import reader
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # or 'WARNING'
import numpy as np
import math
from PIL import Image
import seaborn as sns
import warnings
import random
import datetime
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error , mean_absolute_percentage_error
from tensorflow.keras.losses import Huber
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, BatchNormalization,LeakyReLU
from scipy.stats import pearsonr, linregress
from tqdm import tqdm
from colorama import *
from termcolor import colored
import json
import cv2
import pandas as pd
#==============================
from PyQt5 import QtCore, QtGui, QtWidgets, uic  
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QMessageBox, QFileDialog, QListWidget
import PyQt5.QtWidgets 
#============================= for 3D visualization
from mpl_toolkits.mplot3d import Axes3D
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import math as m
#==============================
cur_path           = os.path.dirname(os.path.abspath(__file__))       # current absolute path 
os.chdir(cur_path)                                                    # change the working path to current working dir

#===============================
from _core import *
from helper_tools import *
#===============================
import webbrowser

#Load UI
Ui_MainWindow, QtBaseClass = uic.loadUiType("ui_interface.res")

class BaseClass(QtWidgets.QMainWindow ):                              #  (QtWidgets.QMainWindow, Ui_MainWindow):            
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        loadUi("ui_interface.res",self)


    def initalize_vars(self):
        self.loadImage(target_img ="logo") 
        self.loadImage(fname ="./assets/about_3dccs.jpg", target_img = "3dccs_help")
        self.sample_img = "./assets/sample_projection.png"
        self.loadImage(self.sample_img  , target_img ="org_mol") 
        self.loadImage(self.sample_img  , target_img ="mod_mol") 
        self.loadImage(self.sample_img  , target_img ="inf_org_mol") 
        self.loadImage(self.sample_img  , target_img ="inf_mod_mol") 
        self.run_mode = "Idle"
        self.start_time = time.time()
        self.cur_train_res = self.image_dim_slider.value()                                                            
        self.update_progressBar()
        self.enable_3d_preview_flag.stateChanged.connect(self.update_gui_vars) 
        self.btn_train_3dcnn.clicked.connect(self.confrim_train_3cnn)
        self.btn_train_3dcnn_kfold.clicked.connect(self.confrim_train_3cnn_kfold)
        self.use_gui_thresholding_flag.stateChanged.connect(self.set_pixel_thresholding_value)

        #=========
        self.zoom_dial.valueChanged.connect(lambda mode_inference: self.preview_img_binarization(False)) 
        self.pixel_threshold_value.valueChanged.connect(lambda mode_inference: self.preview_img_binarization(False)) 
        
        #=========
        self.list_optimized_mols.setStyleSheet("QListWidget { background-color: lightyellow; }")
        self.projection_2d_list.setStyleSheet("QListWidget  { background-color: lightyellow; }")
        self.help3dccs.setStyleSheet("QTextBrowser          { background-color: lightgrey; }")
        #=========

        self.btn_set_train_msdata_csv.clicked.connect(self.set_train_msdata_csv)
        self.btn_set_train_2d_projection_dirpath.clicked.connect(self.set_train_2d_projection_dirpath)

        self.btn_set_SMILE_datafile.clicked.connect(self.set_SMILE_datafile)
        self.btn_set_opt_mol_output_folder.clicked.connect(self.set_opt_mol_output_folder)

        self.btn_set_train_model_path.clicked.connect(self.set_train_model_path)
        self.btn_set_eval_dirpath.clicked.connect(self.set_eval_dirpath)
        self.btn_set_base_model_path.clicked.connect(self.set_base_model_path)

        #========= Inference
        self.btn_set_inference_msdata.clicked.connect(self.set_inference_msdata_path)
        self.btn_set_inference_projection_path.clicked.connect(self.set_inference_projection_path)
        self.btn_load_trained_model.clicked.connect(self.load_trained_model_for_inference)
        self.btn_inference.clicked.connect(lambda post_train_evalulation: self.inference_3dccs(False) )  # lambda var_to-change: targetfunctuion (values)
        #=====================
        self.btn_exit_inference.clicked.connect(self.exit_3dccs)
        self.btn_exit_trainer.clicked.connect(self.exit_3dccs)
        self.btn_export_inf_results.clicked.connect(self.export_table)
      

        self.btn_set_sdf_input_dirpath.clicked.connect(self.set_sdf_input_dirpath)
        self.btn_set_2d_proj_output_dirpath.clicked.connect(self.set_2d_proj_output_dirpath)

        self.btn_optimize_molecule.clicked.connect(self.optimize_molecule)
        self.list_optimized_mols.itemDoubleClicked.connect(self.preview_optimized_structure)

        self.btn_process_2d_projection.clicked.connect(self.projection_2d)
        self.projection_2d_list.itemDoubleClicked.connect(self.preview_2d_projections)

        self.gpu_index, self.gpu_name = self.select_gpu(init = True) #Call the function to get the GPU index and name of GPU device firs time
        self.preview_img_binarization()                              # activate preview
        self.preview_img_binarization(lambda mode_inference: self.preview_img_binarization(True))                       # activate preview for inference as well
        #=======================
        self.btn_goto_repository.clicked.connect(self.open_online_repository)
        #======================= system verbosty
        self.verbosity           = int(self.sys_verbosity.currentText().split("::")[0])


    # Module for updating all GUI-based variables before running any task
    def update_gui_vars(self):
        # Hide non-critical Tensorflow messages
        hide_tf_warnings(suppress_msg=True) if self.flag_ignore_sys_warnings.isChecked() else None
        #==========================================Systemverbosity
        self.verbosity           = int(self.sys_verbosity.currentText().split("::")[0])
        self.train_epoch         = self.train_epoch_slider.value()   # 12  # 
        self.learning_rate       = self.learning_rate_slider.value()  # 0.0001 # make the learning date for model    
        self.image_threshold     = self.pixel_threshold_value.value() # 100    # 200-250 set value for the thresthold
        self.batch_size          = self.batch_size_slider.value()     # 32     # batch size Defulat 10
        self.sample_limit        = None  
        self.min_cutoff_weight   = self.min_cutoff_slider.value() if self.flag_min_cutoff.isChecked() else 0     
        self.max_cutoff_weight   = self.max_cutoff_slider.value() if self.flag_max_cutoff.isChecked() else 10000
        #=============================================
        self.relu_alpha         = self.relu_alpha_box.value()
        self.activation_func    = self.cls_optimizer_type.currentText()
        self.loss_func          = self.model_loss_func.currentText()
        self.lr_decay_epoch     = self.lr_decay_freq.value()
        self.lr_decay_frac      = round(self.lr_decay_perc.value()/100,2)           # convert percentage to frequency
        self.use_lr_decay       = self.use_lr_decay_flag.isChecked()
        #================================================
        self.random_seed        = self.random_seed_slider.value() # 1985            # set the seed for tensoeflow and rnadom np 
        self.use_thresthold_gui = self.use_gui_thresholding_flag.isChecked()        # True by default
        self.set_precision      = self.set_precision_slider.value() #               # set precision for calculation
        self.exp_counter        = self.exp_count.value()
        self.use_multipixel_flag = self.use_multipixel.isChecked()                  # for multi pixels
        self.model_summary_flag  = self.display_model_summary_flag.isChecked()      # shoet the model summary
        self.sort_dataset_flag   = self.dataset_sorting_flag.isChecked()            # sort dataset during initial reading
        self.autoset_expccs_flag = self.check_expccs_flag.isChecked()               # for checking if the exp-ccs exists in the inference mode

        #====================== TRAINING HYPER PARAMETERS
        # Load the dataset and CSV file paths
        if self.post_train_evalulation == True:                                                                                 # for training purpose only
            self.img_dim           = self.image_dim_slider.value()                                                              # for the image resoultion
            self.dataset_path      = self.train_2d_projection_dirpath.toPlainText()                                             #  os.path.join(cur_path, "datasets/dataset_MpHp/")
            self.csv_file_path     = self.train_msdata.toPlainText()                                                            # os.path.join(cur_path, "datasets/high_confidence_MpHp_compounds.csv")
        else:
            self.img_dim         = int(self.inf_img_dim.currentText())  # 32 default
            self.dataset_path    = self.inference_projection_path.toPlainText()                                                 # in case of manual inference
            self.csv_file_path   = self.smile_msdata_filepath.toPlainText()                                                     # in case of manual inference
        #================================================

        self.Base_3dccs_model  = self.base_model_filepath.toPlainText()                                                     # Base model
        self.basemodel_flag    = self.use_basemodel_flag.isChecked()                                                        # fLAG FOR BASE MODEL 
        self.eval_dirpath      = self.train_evalulation_dirpath.toPlainText()                                               # get name of Eval Dirpath
        self.inf_filepath      = self.smile_msdata_filepath.toPlainText()  
        self.store_weights     = self.trained_model_dirpath.toPlainText()                                                    # path to store only final weighst &configs                                                        .to
        self.inf_adduct_type   = self.inf_adduct_info.currentText()
        self.train_adduct_type = self.train_adduct_info.currentText()


        #=================================================
        self.train_singlemode = self.train_singlemode_flag.isChecked()                                                      # train just one dataset as show in th molecular dataset bar

        self.dataset_id        =  os.path.splitext(os.path.basename(self.csv_file_path))[0]
        self.config_fname      =  "_".join( ["config", self.dataset_id ]) +".json" 
        if self.post_train_evalulation == True:
            self.eval_dirname      =  "_".join( [self.dataset_id ,f"{self.cur_train_res}x{self.cur_train_res}_" ])          # Eval dirname is based on MSdata CSV file    
        else:
            self.eval_dirname      =  "_".join( [self.dataset_id ,f"{self.img_dim }x{self.img_dim}_" ])
        self.config_file_path      =  "/".join(["./configs" , self.config_fname])
        #========================
        if os.path.isfile(self.Base_3dccs_model) and  self.use_basemodel_flag.isChecked():
            self.base_reg_model = self.Base_3dccs_model
        else:
            self.base_reg_model = os.path.join(cur_path, "models/regression_model_checkpoint_best_%sx%spxl.h5"%(str(self.img_dim),str(self.img_dim))) 
        #========================
        #self.reg_model_fname   = os.path.join(cur_path, "models/" + self.dataset_id + "_3dccs_model_ckpt_%sx%spxl_%s.h5"%(str(self.img_dim),str(self.img_dim),  str(self.random_seed)))
        self.model_config      = os.path.join(cur_path, "models/" + self.dataset_id + "_3dccs_model_ckpt_%sx%spxl_%s.cfg"%(str(self.img_dim),str(self.img_dim), str(self.random_seed)))
        self.SMILE_src_filepath = self.raw_smile_datafile.toPlainText()
        self.optimized_struct_output_dirpath  = self.set_optimized_mol_datapath.toPlainText()
        self.sdf_mol_dirpath   = self.sdf_mol_input_dirpath.toPlainText()
        self.projection_output = self.projection_output_dirpath.toPlainText()
        self.use_computed_mass = self.use_computed_exact_mass_flag.isChecked()
        #===================================
        self.AllCCSID_name_flag = self.use_AllCCSID_flag.isChecked()
        self.molecule_name_flag = self.use_molecule_name_flag.isChecked()
        #===================================
        self.val_mae_lst        = []                               # list for storing or accessing the absoulte mean validation error 
        self.val_mape_lst       = []

        self.found_sample_names = []                               # holds the anmes of real sampels names in dataset
        self.val_abs_err_lst    = []                               # holds the values for absolute valiadtio nerror list
        self.avgabs_per_err_lst = []                               # absolute validation percentage error
        self.best_val_accuracy  = 10000                            # A validation accuracy (start with high huge random number)
        self.min_perc_error     = 10000                            # start with a random huge number
        #==================================
        self.num_rotation       = self.num_rotation_slider.value() # 5 Default 


        if  self.use_thresthold_gui == True:
            print(colored("#Image binerization set using GUI : %d"%self.image_threshold ,"green"))
        else:
            print(colored("#Using default image threshold    : %d"%self.image_threshold,"green"))
        #==========================
        hide_tf_warnings(suppress_msg=self.flag_ignore_sys_warnings.isChecked()) if self.flag_ignore_sys_warnings.isChecked() else None
        self.gpu_index, self.gpu_name = self.select_gpu(init = False) #Call the function to get the GPU index and name of GPU device 

        # Create directory for evaluation images
        self.timestamp       = datetime.datetime.now().strftime('%Y%m%d%H%M%S')        # create time_stamp for the folder to store th model & associated images
        #====================================================
        self.evaluations_dir = os.path.join(self.eval_dirpath , self.eval_dirname  + self.timestamp)
        #===========================================
        self.set_pixel_thresholding_value()
        #====================================================
        plt.ion()
        self.make_plot()
        #====================================================
        print(colored("--------------------------------" , "green"))
        print(colored(f"Status:  {self.run_mode}" ,"white"))
        print(colored("--------------------------------" , "green"))


    def save_json_config(self, variables, file_path):
        with open(file_path, 'w') as config_file:
            json.dump(variables, config_file, indent=4)

    def load_json_config(self, file_path):
        with open(file_path, 'r') as config_file:
            return json.load(config_file)

    def set_opt_mol_output_folder(self):
        dirname = str(QFileDialog.getExistingDirectory(self, "Select output folder path SDF and molecular co-ordinates (create New Folder if necessary)", os.getcwd()))       
        if dirname:
            self.set_optimized_mol_datapath.setText(dirname)                                        # Set Dirname as path where images are located
            self.sdf_mol_input_dirpath.setText(dirname)  if self.set_autoset_2D_projection_flag.isChecked() else None 
        else:
            None


    #exports the inference results table to the csv file
    def export_table(self):

        if self.tableWidget.rowCount()<=1:
            self.qm.critical(self, "Deep3DCCS", "Error! inference table is empty. Run inference | prediction first")
            return
            
        if self.qm.question(self,'Deep3DCCS',f"Export the prediction results as csv datafile", self.qm.Yes | self.qm.No) == self.qm.No:
            return
        
        # make result directory
        os.makedirs("./results",exist_ok= True)
        # Open Save File dialog
        file_path, _ = QFileDialog.getSaveFileName(self,
            "Export inference result as CSV file",
            os.path.join("./results", os.path.basename(self.smile_msdata_filepath.toPlainText()).split(".")[0] +"_pred.csv"),
            "CSV Files (*.csv);;All Files (*)"  )

        if not file_path:
            return  # User cancelled

        try:
            with open(file_path, mode='w', newline='', encoding='latin-1') as file:
                writer = csv.writer(file)

                # Write header row
                headers = []
                for column in range(self.tableWidget.columnCount()):
                    header_item = self.tableWidget.horizontalHeaderItem(column)
                    headers.append(header_item.text() if header_item else "")
                writer.writerow(headers)

                # Write table data
                for row in range(self.tableWidget.rowCount()):
                    row_data = []
                    for column in range(self.tableWidget.columnCount()):
                        item = self.tableWidget.item(row, column)
                        row_data.append(item.text() if item else "")
                    writer.writerow(row_data)

            print(colored("Inference | Prediction results stored to:", "white"), colored(f"{file_path}" , "green"))
            self.qm.critical(self, "Deep3DCCS", "SUCCESS: Prediction results stored successfully")
            return

        except Exception as e:
            print("Error saving prediction results:", e)


    def load_trained_model_for_inference(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Select trained models","./models", "HDF5 model weights (*.h5);;All Files (*.*)", options=options)
        if fileName:
            self.trained_model.setText(fileName)
        else:
            None

    def open_online_repository(self):
        try:
            webbrowser.open("https://github.com/vivekmathema/Deep3DCCS")
        except:
            print("[WARNING] Falied to open web browser")
            pass




    def set_inference_msdata_path(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Select raw SMILEs input datafile","./datasets", "SMILEs RAW datafile (*.csv);;All Files (*.*)", options=options)
        if fileName:
            self.smile_msdata_filepath.setText(fileName)
            if self.qm.question(self,'Deep3DCCS',"Automatically set the 2D projection folder path", self.qm.Yes | self.qm.No) == self.qm.No:
                return
            self.inference_projection_path.setText(os.path.join(get_pathname_without_ext(fileName) + "_optimized_structure", "2D_projections")) # autoset path according to the source data file
        else:
            None

    def set_inference_projection_path(self):
        dirname = str(QFileDialog.getExistingDirectory(self, "Select 2D Projections folder of corrsponding SMILEs data", "./datasets"))       
        if dirname:
            self.inference_projection_path.setText(dirname)                                        # Set Dirname as path where images are located
        else:
            None

    #----------------------------

    def set_train_msdata_csv(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ =  QFileDialog.getOpenFileName(self,"Select raw SMILEs input datafile","./datasets", "SMILEs RAW datafile (*.csv);;All Files (*.*)", options=options)
        if fileName:
            self.train_msdata.setText(fileName)            
            if self.qm.question(self,'Deep3DCCS',"Automatically set the 2D projection folder path", self.qm.Yes | self.qm.No) == self.qm.No:
                return
            self.train_2d_projection_dirpath.setText(os.path.join(get_pathname_without_ext(fileName) + "_optimized_structure", "2D_projections")) # autoset path according to the source data file
        else:
            None

    def set_train_2d_projection_dirpath(self):
        dirname = str(QFileDialog.getExistingDirectory(self, "Select folder path for 2D projection training data", os.path.join(cur_path,"datasets"))) 
        if dirname:
            self.train_2d_projection_dirpath.setText(dirname)                                        # Set Dirname as path where images are located
        else:
            None

    def set_base_model_path(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Select base model","3DCSS ","All Files (*.*);;Image Files (*.h5,*.model)", options=options)
        if fileName:
            self.base_model_filepath.setText(fileName)
        else:
            None


    def set_train_model_path(self):
        dirname = str(QFileDialog.getExistingDirectory(self, "Select folder path for traiend model", os.path.join(cur_path,"models"))) 
        if dirname:
            self.trained_model_dirpath.setText(dirname)                                         # Set Dirname as path where images are located
        else:
            None

    def set_eval_dirpath(self):
        dirname = str(QFileDialog.getExistingDirectory(self, "Select folder path for training evaluation results", os.path.join(cur_path,"models"))) 
        if dirname:
            self.train_evalulation_dirpath.setText(dirname)                                         # Set Dirname as path where images are located
        else:
            None

    def set_SMILE_datafile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        fileName, _ = QFileDialog.getOpenFileName(
            self,
            "Select SMILE data file",
            os.path.join(cur_path, "datasets"),
            "Data files (*.csv *.xls *.xlsx);;CSV files (*.csv);;Excel files (*.xls *.xlsx);;All Files (*.*)",
            options=options
        )

        if fileName:
            self.raw_smile_datafile.setText(fileName)


    def set_sdf_input_dirpath(self):
        dirname = str(QFileDialog.getExistingDirectory(self, "Select input folder path for SDF and molecular co-ordinates", os.getcwd())) 
        if dirname:
            self.sdf_mol_input_dirpath.setText(dirname)                                        # Set Dirname as path where images are located
        else:
            None

    def set_2d_proj_output_dirpath(self):
        dirname = str(QFileDialog.getExistingDirectory(self, "Select output folder path for 2d Projection images (create New Folder if necessary)", os.getcwd()))       
        if dirname:
            self.projection_output_dirpath.setText(dirname)                                        # Set Dirname as path where images are located
        else:
            None

    def optimize_molecule(self):
        if self.qm.question(self,'Deep3DCCS',"Process SMILEs for 3D molecule structure optimization?", self.qm.Yes | self.qm.No) == self.qm.No:
            return
        if self.optimized_struct_output_dirpath.strip() == "":
            print(colored("ERROR! Output path is empty. Please seelct a path (is created if not exists)" ,"red"))
            return
        else:
            os.makedirs(self.optimized_struct_output_dirpath, exist_ok = True)  

        #===== Progress bar stuffs
        self.run_mode   = "3D structure optimization and SDF file construction"
        self.start_time = time.time()
        #=====
        self.update_gui_vars()
        self.dataset_id = os.path.splitext(os.path.basename(self.SMILE_src_filepath))[0]                      # during the dataset construction
        self.set_gpu_memory_growth(gpu_index = self.gpu_index, set_flag = self.set_gpu_growth.isChecked())    # set GPU mem growth
        self.console_show_store_vars()
        self.build_otptimized_structure_from_smile(input_smile_datafile = self.SMILE_src_filepath)            # call moel optimization


    def projection_2d(self):
        if self.qm.question(self,'Deep3DCCS',"Process SMILES-based optimized 3D structures for 2D projection dataset construction?", self.qm.Yes | self.qm.No) == self.qm.No:
            return
        if self.projection_output.strip() == "":
            print(colored("ERROR! Output path is empty. Please select a path (will be created if not exists)" ,"red"))
            return
        else:
            os.makedirs(self.optimized_struct_output_dirpath, exist_ok = True) 
            #os.makedirs(self.sdf_mol_dirpath, exist_ok = True)                                                      # make directory if required

        #===== Progress bar stuffs
        self.run_mode   = "2D projection dataset construction"
        self.start_time = time.time()
        #=====
        self.update_gui_vars()
        print(colored(f"Number of angle rotations: {self.num_rotation}", "green"))
        self.set_gpu_memory_growth(gpu_index = self.gpu_index, set_flag = self.set_gpu_growth.isChecked())    # set GPU mem growth
        self.console_show_store_vars()
        self.build_2D_projections(sdf_mol_dirpath = self.sdf_mol_dirpath)            # call moel optimization
    
    def preview_optimized_structure(self):
        if self.preview_optimize_item_flag.isChecked():
            self.show_preview_structure =  os.path.join(self.struct_output_dir, self.list_optimized_mols.currentItem().text()+ "_optimized_structure.sdf")
            print(colored(f"Showing optimized structure  : { shorten_path(self.show_preview_structure) }", "yellow") )
            self.show_preview_molecules(self.show_preview_structure)

    def preview_2d_projections(self):
        if self.preview_2d_projection_flag.isChecked():
            self.sample_projection_preview_path = os.path.join(self.projection_output, self.projection_2d_list.currentItem().text())
            self.sample_projection_preview_path = os.path.splitext(self.sample_projection_preview_path)[0]

            #self.sample_projection_preview_path = "./datasets/AllCCS_UnifiedCCSlevel1_MpHp_optimized_structure/2D_projections/1-Methyltryptamine_coordinates"

            print(colored(f"Showing 2D projections : { shorten_path(self.sample_projection_preview_path) }", "yellow") )
            self.show_2d_projections_preview(projection_dirpath = self.sample_projection_preview_path )


    def update_progressBar(self,tot_index =0 , cur_index = 0):
        if self.run_mode == "Idle":
            self.progressBar_state.setText("network idle: awaiting task")
            return
        ETA=(tot_index - cur_index) * (time.time()-self.start_time)/(2*60)
        itm_per_sec =  2/(0.0001 + time.time()-self.start_time)                        # rough ETA where 0.0001 is epselon for time.time()
        self.start_time = time.time()
        self.progressBar.setValue(round(100 * (cur_index +1 ) /tot_index) )
        self.progressBar_state.setText("status: %s | ETA(min): %0.5f | ~(epoch, iter., items)/s : %0.4f |"%(self.run_mode, ETA, itm_per_sec) ) 


    def loadImage(self,fname= "./assets/logo.png" , target_img = "logo" ):
        self.image=cv2.imread(fname,cv2.IMREAD_COLOR)

        if not target_img ==  "3dccs_help":                               # don't resize for the Help menu
            self.image = image_resize(self.image, height = 100)

        try:
            self.app_logo.setPixmap(QPixmap.fromImage(self.displayImage(self.image))) if target_img ==  "logo" else None
            self.app_logo.setAlignment(QtCore.Qt.AlignCenter)                         if target_img ==  "logo" else None

            self.mol_orginal.setPixmap(QPixmap.fromImage(self.displayImage(self.image))) if target_img ==  "org_mol" else None
            self.mol_orginal.setAlignment(QtCore.Qt.AlignCenter)                         if target_img ==  "org_mol" else None

            self.mol_modified.setPixmap(QPixmap.fromImage(self.displayImage(self.image))) if target_img ==  "mod_mol" else None
            self.mol_modified.setAlignment(QtCore.Qt.AlignCenter)                         if target_img ==  "mod_mol" else None

            self.flowchart_3dccs.setPixmap(QPixmap.fromImage(self.displayImage(self.image))) if target_img ==  "3dccs_help" else None
            self.flowchart_3dccs.setAlignment(QtCore.Qt.AlignCenter)                         if target_img ==  "3dccs_help" else None

        except Exception as e:
            print(f"Preview update (or load) {target_img} skipped due to potential memory error.. {e}")

    def displayImage(self, img):                                                     # Returns a qimage, must eb a colour image
        qformat =QImage.Format_Indexed8
        if len(img.shape)==3:
            if img.shape[2] ==4:
                qformat=QImage.Format_RGBA8888
            else:
                qformat=QImage.Format_RGB888
            img = QtGui.QImage(img.data,
                img.shape[1],
                img.shape[0], 
                img.strides[0], qformat)
            img = img.rgbSwapped()
        return img

    def set_pixel_thresholding_value(self):
        if  self.use_gui_thresholding_flag.isChecked():
            self.tabMultiTaskWIdget.setCurrentIndex(2)
            self.image_threshold    = self.preview_img_binarization()
            print(colored(f"#Updated Binarization thresholkd value : {self.image_threshold}", "green" ))
        else:
            pass

    def zoom_mol_config(self):
        self.lcdNumber.setValue(self.zoom_dial.value())

    def preview_img_binarization(self, mode_inference =False):  # makes the thresathold of image ( higehr the img_dim, teh betetr but huge memory requirments)
        self.loadImage(self.sample_img, target_img ="org_mol")             # Display the two-bit image (new_threshold_value = 0, )

        def getImage(img):
            if len(img.shape) == 2:  # Grayscale image
                qformat = QtGui.QImage.Format_Indexed8
                qimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.strides[0], qformat)
            elif len(img.shape) == 3:  # Color image
                if img.shape[2] == 4:
                    qformat = QtGui.QImage.Format_RGBA8888
                else:
                    qformat = QtGui.QImage.Format_RGB888
                qimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], img.strides[0], qformat)
                qimg = qimg.rgbSwapped()
            return qimg


        new_threshold_value = self.pixel_threshold_value.value() 

        preview_zoom_dim = int(300 + self.zoom_dial.value())   

        # Read and resize the image (for modified image)
        self.image = cv2.imread(self.sample_img , cv2.IMREAD_COLOR)        
        # Convert to grayscale
        gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)                                                    # Read image 
        gray_image = cv2.resize(gray_image, (preview_zoom_dim, preview_zoom_dim), interpolation=cv2.INTER_AREA)      # Resize image          
        image_array = np.array(gray_image)

        # For modified image                                                                           # convert to array                        
        two_bit_array_mod = (image_array > new_threshold_value).astype(np.uint8)                       # Binarize the image
        two_bit_image_mod = Image.fromarray(two_bit_array_mod * 255)
    
        self.mol_modified.setPixmap(QtGui.QPixmap.fromImage(getImage(two_bit_array_mod * 255)))        # Dipslay updated image on on QPixmap 
        self.mol_modified.setAlignment(QtCore.Qt.AlignCenter)                                          # Align the image

        # for Orginal image
        two_bit_array_org = (image_array > 0 ).astype(np.uint8)                                                      # no chnage
        two_bit_image = Image.fromarray(two_bit_array_org * 255)               
        

        self.mol_orginal.setPixmap(QtGui.QPixmap.fromImage(getImage(two_bit_array_org * 255)))  # Dipslay updated image on on QPixmap 
        self.mol_orginal.setAlignment(QtCore.Qt.AlignCenter)

        return new_threshold_value # Return the threshold value set by the user


    def select_gpu(self, init = False):    
        
        if init == True:                                                            # Initalizing the GPU scan on start  
            #print(colored("#TensorFlow version:%s"%str(tf.__version__), "blue"))   # Check if TensorFlow is installed and print its version
            self.use_processor.clear()                                              # Clear GPU

            gpus = tf.config.experimental.list_physical_devices('GPU')              # Check if CUDA (GPU support) is available and print GPU information
            if gpus and tf.test.is_built_with_cuda():                               # make sure the tf is built with cuda
                print("#CUDA is available. Listing GPUs...")
                print("_________________________________________________\n")
                available_gpu_names = [gpu.name for gpu in gpus]                    # List available GPUs and their names
                for i, gpu_name in enumerate(available_gpu_names):                  # add GPUs and CPUs to teh list
                    self.use_processor.addItem(f"{i}::GPU--{gpu_name}")
                    print(colored(f"[ GPU {i}::GPU--{gpu_name}", "green"))
                self.use_processor.addItem("-1::CPU--CPU")                          # By default ass CPU at end
                print("_________________________________________________")
                return 0, gpus[0].name
            else:                                                                   # only add CPU and resturn
                print("#CUDA is not available. Defaulting to CPU. Processing will be very slow...")
                self.use_processor.addItem("0::CPU--CPU")
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
                return "/CPU:0", "CPU"

        #self.use_processor.setCurrentIndex(0)
        gpu_index =self.use_processor.currentText().split("::")[0]          # get GPU index at current selected text
        gpu_name = self.use_processor.currentText().split("--")[1]          # get GPU name at curren tselected text

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)                 # set GPU device for processing 3DCNN tensor operations

        if gpu_index == -1:
            print(colored("Warning! Using CPU. Processing will be very slow...", "red"))
        else:          
            print(colored(f"#GPU processor   : {gpu_index} | {gpu_name}", "yellow"))

        return gpu_index, gpu_name     


    def set_gpu_memory_growth(self, gpu_index=0, set_flag=False): 
        # Set the GPU index
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
        physical_devices = tf.config.list_physical_devices('GPU') 
        for gpu_instance in physical_devices:
            try:     
                tf.config.experimental.set_memory_growth(gpu_instance, set_flag)
            except:
                print(colored(f"Error setting GPU memory for : {gpu_instance}", "red"))
                pass


