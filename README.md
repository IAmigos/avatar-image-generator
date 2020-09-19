# Avatar Image Generator
  
   Faces Domain <br/>
   <img src="images/Faces_example.jpeg" width="500" />
   
   Cartoons Domain<br/>
   <img src="images/Cartoons_example.jpeg" width="500" />

   Based on the paper: https://arxiv.org/abs/1711.05139

## The problem
  
This repo aims to contribute to the daunting problem of generating a cartoon given the picture of a face.  <br/> <br/>
This is an image-to-image translation problem, which involves many classic computer vision tasks, like style transfer, super-resolution, colorization and semnantic    segmentation. Also, this is a many-to-many mapping, which means that for a given face there are multiple valid cartoons, and for a given cartoon there are multiple valid faces too. </br>
  
## Datasets 

  Faces dataset: we use the VggFace dataset (https://www.robots.ox.ac.uk/~vgg/data/vgg_face/) from the University of Oxford
  
  Cartoon dataset: we use the CartoonSet dataset from Google (https://google.github.io/cartoonset/), both the versions of 10000 and 100000 items. 

## The repo
  
  datasets -> </br>
          /face_datasets/face_images_wo_bg -> well-segmented selected faces from the vggface dataset </br>
          /face_datasets/face_images_wo_bg_permissive -> segmented faces from the vggface dataset, being permissive on the quality of the segmentation when selecting the images </br>
          /face_datasets/vgg_face_dataset -> vggface dataset as it is </br>
          /cartoon_datasets/cartoonset10k -> 10k cartoon dataset from CartoonSet </br>
          /cartoon_datasets/cartoonset100k_limited -> about 16k cartoon dataset from CartoonSet, choosing cartoons so that some features are avoided, like sunglasses, and dataset-shift problem is mitigated </br>
  images -> sample images just for presentation</br>
  xgan_model.ipynb -> contains the current model for the generation of avatars from faces (.py version soon to come)</br>
  face_segmentation.ipynb -> notebook containing the segmentation model for the faces (.py version soon to come) </br>
  
## The model

   It is based on the XGAN paper, omitting the Teacher Loss, and adding an autoencoder in the end, trained to learn well only the representation of the cartoons, as to "denoise" the spots and wrong colorisation from the face-to-cartoon outputs of the XGAN. 



