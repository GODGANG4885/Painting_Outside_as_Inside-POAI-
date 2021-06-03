# Painting Outside as Inside (POAI): Edge guided image outpainting via bidirectional rearrangement with progressive step learning

<!-- You can use the [editor on GitHub](https://github.com/GODGANG4885/Painting_Outside_as_Inside-POAI-/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files. -->
 <center>Kyunghun Kim, Y.hun Yun, K.Woo Kang, K.bo Kong, S.yeong Lee, and Suk-Ju Kang</center>  
 
 <center>  Sogang University  POSTECH  NAVER LABS</center>  
 
##  <center> in WACV 2020 (Oral)</center>  


![fig1](https://user-images.githubusercontent.com/36159663/120644590-77ea5200-c4b2-11eb-9e39-45ba369f36c5.png)

## Abstract  

Image outpainting is a very intriguing problem as the outside of a given image can be continuously filled by considering as the context of the image. This task has two main
challenges. The first is to maintain the spatial consistency in contents of generated regions and the original input. The second is to generate a high-quality large image with a small amount of adjacent information. Conventional image outpainting methods generate inconsistent, blurry, and repeated pixels. To alleviate the difficulty of an outpainting problem, we propose a novel image outpainting method using bidirectional boundary region rearrangement. We rearrange the image to benefit from the image inpainting task by reflecting more directional information. The bidirectional boundary region rearrangement enables the generation of the missing region using bidirectional information similar to that of the image inpainting task, thereby generating the higher quality than the conventional methods using unidirectional information. Moreover, we use the edge map generator that considers images as original input with structural information and hallucinates the edges of unknown regions to generate the image. Our proposed method is compared with other state-of-the-art outpainting and inpainting methods both qualitatively and quantitatively. We further compared and evaluated them using BRISQUE, one of the No-Reference image quality assessment (IQA) metrics, to evaluate the naturalness of the output. The experimental results demonstrate that our method outperforms other methods and generates new images with 360°panoramic characteristics.

## Proposed Method
![model2](https://user-images.githubusercontent.com/36159663/120647203-4b840500-c4b5-11eb-91df-1053f4e71ceb.png)

## <center> Paper </center>
