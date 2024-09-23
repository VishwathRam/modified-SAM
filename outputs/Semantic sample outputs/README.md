# Understanding the sample outputs

It is important to note the following points:<br><br>

- SAM gives class agnostic outputs, so it doesn't have class information to seperate between different objects. Rather it tries to find some structure within the whole image and segments that. So do note that on combining this with a class-separator model we _might_ get even better outputs.
- SAM is built as a promptable model. It performs best when fed with relevant prompts. The model we have trained is trained to work with
  bounding box prompts (as is sensible for the case of polygonal annotation). However, such a tool to select bounding boxes is not immediately available to me, and so I have prompted bounding boxes as basically the entire image. So do note that giving appropriate bounding boxes for different parts of the image _might_ lead to even better results.
- SAM has the capability to produce multiple possibilities for segment masks through its mask decoder module. This can be helpful for segmenting different objects or assimilating information from all the mask predictions into one single superior mask prediction. We have so far only explored single output masks. So do note that exploring multi-mask outputs _might_ lead to even better results.
- SAM is built for 1024x1024 image inputs. Hence in our prediction pipeline, image inputs with a different size are currently directly resized to this regardless of aspect ratio. However, it is generally accepted in the ML community that vision models are sensitive to aspect ratios and changing them might lead to worse outputs. Moreover, distorting aspect ratios has an even greater impact on edge representation and might have a greater effect on dichotomous segmentation. So do note that exploring aspect-ratio maintenance _might_ lead to even better results.
- SAM is trained with an image processor that by default normalizes images to certain standard values. In our experiments we observed that for some images this might render some areas completely dark and indistinguishable even to the human eye and unsuitable for edge detection by the model. For this reason, we performed training on dichotomous images without normalization. So do note that on exploring normalization and finding a better balance for normalization and preprocessing _might_ lead to even better results.
- In addition to binary mask outputs overlayed on images, probability heatmaps of the model outputs have also been included here. These denote the probabilities of each pixel belonging to a mask or not, without filtering them to a binary mask. 

<br>
Besides, all of these outputs were fed to the prediction module on default settings, tinkering around with some thresholds and params might lead to better results.<br>
Some other potential modifications that might lead to even better results can be discussed at length separately.
That said, it is important to realize the model might not work just as well on all sorts of images, after all ML is an iteratively improving effort!
