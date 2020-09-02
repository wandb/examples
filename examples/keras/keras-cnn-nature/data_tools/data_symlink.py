import argparse
import json
import os
import pickle
import random
from shutil import copyfile

# Useful subsets of the taxonomy
#-------------------------------
# proper classes with the most example images
top_taxa = ["Amphibia", "Aves", "Insecta", "Mammalia", "Reptilia"] 
# species with the most examples, 5 from each of above classes
top_species = ['Sceloporus occidentalis', 'Trachemys scripta elegans', 'Anolis carolinensis', 'Chelydra serpentina', 'Crotalus atrox', 'Odocoileus virginianus', 'Sciurus niger', 'Sciurus carolinensis', 'Procyon lotor', 'Canis latrans', 'Lithobates catesbeianus', 'Incilius nebulifer', 'Anaxyrus americanus', 'Lithobates clamitans', 'Pseudacris sierra', 'Anas platyrhynchos', 'Ardea herodias', 'Buteo jamaicensis', 'Ardea alba', 'Branta canadensis', 'Danaus plexippus', 'Apis mellifera', 'Pachydiplax longipennis', 'Junonia coenia', 'Vanessa atalanta']
# top 10 taxa overall (including proper classes and kingdoms)
taxa_10 = ["Aves", "Plantae", "Insecta", "Reptilia", "Mammalia", "Amphibia", "Mollusca", "Fungi", "Animalia", "Arachnida"]

# output label config
CLASSES = taxa_10

# fixed path names
#-------------------
train_dir = "train"
val_dir = "val"

# load dictionaries storing image ids (jpg file names) organized by the class/species label 
jpgs_by_class = pickle.load(open('class_to_jpg_file_name.pkl', 'rb'))
jpgs_by_species = pickle.load(open('species_to_jpg_file_name.pkl', 'rb'))

def build_symlink_data(args):
  """
  Build train and validation split in the root directory. For each class, take a random sample of images
  and generate symlinks to them at the correct node of the overall data tree
  root
    train
      Class A
        image0.jpg
        image1.jpg
        ...
      Class B
        image10.jpg
        image11.jpg
        ...
    val
      Class A
        image50.jpg
        image51.jpg
        ...
      Class B
        image60.jpg
        image61.jpg
        ...
  """
  if not os.path.isdir(args.dest_data):
    os.mkdir(args.dest_data)
  os.mkdir(os.path.join(args.dest_data, train_dir))
  os.mkdir(os.path.join(args.dest_data, val_dir))
  
  # save a second copy of the data with the parent class as the prediction label
  # that is, the destination directory args.dest_data will contain training and validation data for 25 species,
  # and a second directory with the suffix _BY_CLASS will contain training and validation data for
  # 5 biological classes, where each class is the parent of 5 of the 25 species 
  if args.flat_target:
    by_class_dest = args.dest_data + "_BY_CLASS"
    os.mkdir(by_class_dest)
    os.mkdir(os.path.join(by_class_dest, train_dir)) 
    os.mkdir(os.path.join(by_class_dest, val_dir)) 
    for parent_class in top_taxa:
      os.mkdir(os.path.join(by_class_dest, train_dir, parent_class))
      os.mkdir(os.path.join(by_class_dest, val_dir, parent_class))

    # the limited number of species for which we have enough data (25) is hardcoded as top_species, each corresponding
    # to a parent class in top_taxa, such that each class has 5 constituent species
    # when training to predict class and species, save each selected file twice
    # (once for class data tree, once for species data tree)   
    for s in top_species:
      os.mkdir(os.path.join(args.dest_data, train_dir, s))
      os.mkdir(os.path.join(args.dest_data, val_dir, s))

      jpgs = random.sample(jpgs_by_species[s], args.train_count + args.val_count)
      # validation 
      for j in jpgs[:args.val_count]:
        curr_class, species, filename = j.split("/")
        src_img = os.path.join(args.src_data, j)
        if args.copy_real_files:
          copyfile(src_img, os.path.join(args.dest_data, val_dir, species, filename))
          copyfile(src_img, os.path.join(by_class_dest, val_dir, curr_class, filename))
        else:
          os.symlink(src_img, os.path.join(args.dest_data, val_dir, species, filename))
          os.symlink(src_img, os.path.join(by_class_dest, val_dir, curr_class, filename))
      # training
      for j in jpgs[args.val_count:]:
        curr_class, species, filename = j.split("/")
        src_img = os.path.join(args.src_data, j)
        if args.copy_real_files:
          copyfile(src_img, os.path.join(args.dest_data, train_dir, species, filename))
          copyfile(src_img, os.path.join(by_class_dest, train_dir, curr_class, filename))
        else:
          os.symlink(src_img, os.path.join(args.dest_data, train_dir, species, filename))
          os.symlink(src_img, os.path.join(by_class_dest, train_dir, curr_class, filename))

      num_val_files = len(os.listdir(os.path.join(args.dest_data, val_dir, s)))
      num_train_files = len(os.listdir(os.path.join(args.dest_data, train_dir, s)))
      print(s, ": ", num_train_files, " train, ", num_val_files, " val")


  # when training to predict classes alone, take a random sample of N_TRAIN + N_VAL size from each class
  else:
    for c in CLASSES:
      os.mkdir(os.path.join(args.dest_data, train_dir, c))
      os.mkdir(os.path.join(args.dest_data, val_dir, c))
    
      jpgs = random.sample(jpgs_by_class[c], args.train_count + args.val_count)
      # validation
      for j in jpgs[:args.val_count]:
        jpg_filename = j.split("/")[-1]
        src_img = os.path.join(args.src_data, c, j)
        if args.copy_real_files:
          copyfile(src_img, os.path.join(args.dest_data, val_dir, c, jpg_filename))
        else:
          os.symlink(src_img, os.path.join(args.dest_data, val_dir, c, jpg_filename))
      # training  
      for j in jpgs[args.val_count:]:
        jpg_filename = j.split("/")[-1]
        src_img = os.path.join(args.src_data, c, j)
        if args.copy_real_files:
          copyfile(src_img, os.path.join(args.dest_data, train_dir, c, jpg_filename))
        else:
          os.symlink(src_img, os.path.join(args.dest_data, train_dir, c, jpg_filename))
    
      num_val_files = len(os.listdir(os.path.join(args.dest_data, val_dir, c)))
      num_train_files = len(os.listdir(os.path.join(args.dest_data, train_dir, c)))
      print(c, ": ", num_train_files, " train, ", num_val_files, " val")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "-s",
    "--src_data",
    type=str,
    help="Absolute path to root of source data")
  parser.add_argument(
    "-d",
    "--dest_data",
    type=str,
    help="Absolute path to root of destination/symlink data")
  parser.add_argument(
    "-t",
    "--train_count",
    type=int,
    help="Number of training examples per class")
  parser.add_argument(
    "-v",
    "--val_count",
    type=int,
    help="Number of validation examples per class")
  parser.add_argument(
    "--two_tiers",
    action="store_true",
    help="Build two tiers of prediction (class and species) instead of flat class hierarchy (species only)")
  parser.add_argument(
    "-c",
    "--copy_real_files",
    action="store_true",
    help="If set, save actual files (not just symlinks)")
  build_symlink_data(parser.parse_args()) 
