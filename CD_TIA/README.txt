A Weakly-supervised Multi-lesion SegmentationFramework Based on Target-level Incomplete Annotations

Code of MICCAI 2024 paper: A Weakly-supervised Multi-lesion SegmentationFramework Based on Target-level Incomplete Annotations

Abastract
Effectively segmenting Crohn's disease (CD) from computed tomography is crucial for clinical use. Given the difficulty of
obtaining manual annotations, more and more researchers have begun to pay attention to weakly supervised methods. However,
due to the challenges of designing weakly supervised frameworks with limited and complex medical data, most existing
frameworks tend to study single-lesion diseases ignoring multi-lesion scenarios. In this paper, we propose a new local-to-global
weakly supervised neural framework for effective CD segmentation. Specifically, we develop a novel weak annotation strategy
called Target-level Incomplete Annotation (TIA). This strategy only annotates one region on each slice as a labeled
sample, which significantly relieves the burden of annotation. We observe that the classification networks can discover
target regions with more details when replacing the input images with their local views. Taking this into account, we
first design a TIA-based affinity cropping network to crop multiple local views with global anatomical information from
the global view. Then, we leverage a local classification branch to extract more detailed features from multiple local
views. Our framework utilizes a local views-based class distance loss and cross-entropy loss to optimize local and global
classification branches to generate high-quality pseudo-labels that can be directly used as supervisory information for
the semantic segmentation network. Experimental results show that our framework achieves an average DSC score of 47.8%
on the CD71 dataset.

Preparations

TIA dataset
We conduct experiments on the CD71 dataset to verify the effectiveness of our framework.
1.The directory sctructure should be
   GClass
      data
         train
           ill
           noill
         vaild
           ill
           noill

   LClass
      data
         train
           ill
           noill
         vaild
           ill
           noill
2.Affinity Cropping dataset
  /LClass/util/AMatrix/README.txt

Create and activate conda environment
   conda create --name py37 python=3.7
   conda activate py37

Train
  To start global classification branch training, run the scripts under GClass/.
      main.py
      cam.py

  To start local classification branch training, run the scripts under LClass/.
      main.py
      cam.py

  To start local classification branch training, run the scripts under Unet/.
      train.py
      predict.py

Results
The generated CAMs and semantic segmentation results on the CD71 dataset.
The model is trained on CD71 dataset. For more results, please see the [Project page] or [Paper].