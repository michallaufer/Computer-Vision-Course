
import json

def update_clip_dino():
    print("Updating CLIP_DINO.ipynb...")
    try:
        with open('CLIP_DINO.ipynb', 'r', encoding='utf-8') as f:
            nb = json.load(f)

        for cell in nb['cells']:
            if cell['cell_type'] == 'markdown':
                source = "".join(cell['source'])
                if "Inline Question 1" in source and "Your Answer" in source:
                    cell['source'] = [
                        "**Inline Question 1** -\n",
                        "\n",
                        "Why does CLIP's learning depend on the batch size? If the batch size is fixed, what strategy can we use to learn rich image features?\n",
                        "\n",
                        "$\\color{blue}{\\textit Your Answer:}$\n",
                        "\n",
                        "CLIP's contrastive loss uses in-batch negatives. Larger batches provide more negative samples, which are essential for learning discriminative features and preventing collapse. If batch size is fixed, strategies like **Memory Banks** or **MoCo (Momentum Contrast)** can be used to maintain a large queue of negative samples from previous batches, decoupling the number of negatives from the mini-batch size.\n"
                    ]
                elif "Inline Question 2" in source and "Your Answer" in source:
                    cell['source'] = [
                        "**Inline Question 2** -\n",
                        "\n",
                        "CLIP learns to align image and text representations in a shared latent space using a contrastive loss. How would you extend this idea to more than two modalities?\n",
                        "\n",
                        "$\\color{blue}{\\textit Your Answer:}$\n",
                        "\n",
                        "To extend to more modalities (e.g., Audio, Video, Text), we can learn a shared embedding space where all modalities are aligned. We can compute pairwise contrastive losses between every pair of modalities (e.g., Image-Audio, Image-Text, Text-Audio) and minimize the sum of these losses. Alternatively, one modality can serve as an 'anchor' (like Image) to which all other modalities are aligned.\n"
                    ]
                elif "Inline Question 3" in source and "Your Answer" in source:
                    cell['source'] = [
                         "\n",
                        "**Inline Question 3**\n",
                        "\n",
                        "How do we get the tensor shapes printed above? Explain your answer.\n",
                        "\n",
                        "\n",
                        "$\\color{blue}{\\textit Your Answer:}$\n",
                        "\n",
                        "The input image tensor is (1, 3, 480, 480). \n",
                        "1. **Input**: (1, 3, 480, 480). \n",
                        "2. **Patches**: The ViT-S/8 model uses 8x8 patches. For a 480x480 image, we have (480/8)*(480/8) = 60*60 = 3600 patches. \n",
                        "3. **Tokens**: The model adds a [CLS] token, so the total number of tokens is 3600 + 1 = 3601. \n",
                        "4. **Embedding**: ViT-S has an embedding dimension of 384. \n",
                        "Thus, `all_tokens` has shape (1, 3601, 384). `patch_tokens` excludes the [CLS] token, resulting in (1, 3600, 384).\n"
                    ]
                elif "Inline Question 4" in source and "Your Answer" in source:
                     cell['source'] = [
                        "**Inline Question 4** -\n",
                        "\n",
                        "What kind of structure do you see in the visualization above? What does it imply when a region consistently appears in a specific color? What does it mean when two regions have distinctly different color? Remember that PCA reveals the directions of highest variance in the feature space across all patches. A patch's color reflects its distinct feature content.\n",
                        "\n",
                        "\n",
                        "$\\color{blue}{\\textit Your Answer:}$\n",
                        "\n",
                         " The visualization shows that the DINO model captures semantic object parts and boundaries. Regions corresponding to the same semantic object (like the dog's body or the background grass) appear in consistent colors, implying they have similar feature representations. Distinctly different colors indicate different semantic regions. This suggests that DINO learns to group semantically related patches without explicit supervision.\n"
                     ]
                elif "Inline Question 5" in source and "Your Answer" in source:
                     cell['source'] = [
                        "**Inline Question 5** -\n",
                        "\n",
                        "If you train a segmentation model on CLIP ViT's patch features, do you expect it to perform better or worse than DINO? Why should that be the case?\n",
                        "\n",
                        "\n",
                        "\n",
                        "$\\color{blue}{\\textit Your Answer:}$\n",
                        "\n",
                        "We expect a model trained on CLIP ViT features to perform **worse** on segmentation than DINO. CLIP is trained with a global image-text contrastive loss, which encourages the [CLS] token to capture global semantic information but does not explicitly enforce spatial consistency or local discriminability of patch tokens. DINO, on the other hand, uses a local-to-global correspondence objective (multi-crop), which encourages the model to learn fine-grained, spatially coherent features that are more suitable for dense prediction tasks like segmentation.\n"
                     ]
        
        with open('CLIP_DINO.ipynb', 'w', encoding='utf-8') as f:
            json.dump(nb, f, separators=(',', ':'))
        print("Updated CLIP_DINO.ipynb")
    except Exception as e:
        print(f"Error updating CLIP_DINO.ipynb: {e}")

def update_transformer_captioning():
    print("Updating Transformer_Captioning.ipynb...")
    try:
        with open('Transformer_Captioning.ipynb', 'r', encoding='utf-8') as f:
            nb = json.load(f)
            
        for cell in nb['cells']:
            source = "".join(cell['source'])
            if cell['cell_type'] == 'markdown':
                if "Inline Question 1" in source and "Your Answer" in source:
                    cell['source'] = [
                        "# Inline Question 1\n",
                        "\n",
                        "Several key design decisions were made in designing the scaled dot product attention we introduced above. Explain why the following choices were beneficial:\n",
                        "1. Using multiple attention heads as opposed to one.\n",
                        "2. Dividing by $\\sqrt{d/h}$ before applying the softmax function. Recall that $d$ is the feature dimension and $h$ is the number of heads.\n",
                        "3. Adding a linear transformation to the output of the attention operation.\n",
                        "\n",
                        "Only one or two sentences per choice is necessary, but be sure to be specific in addressing what would have happened without each given implementation detail, why such a situation would be suboptimal, and how the proposed implementation improves the situation.\n",
                        "\n",
                        "**Your Answer:**\n",
                        "1. **Multiple Heads**: Allows the model to jointly attend to information from different representation subspaces at different positions. A single head would limit the model to one specific type of attention pattern.\n",
                        "2. **Scaling**: Divding by $\\sqrt{d/h}$ counteracts the effect of large dot products growing in magnitude with dimension, which would push the softmax function into regions with extremely small gradients, impeding training.\n",
                        "3. **Linear Transformation**: Mixes the information from the different heads, allowing the model to aggregate the diverse features learned by each head into a unified representation.\n"
                    ]
                elif "Inline Question 2" in source and "Your Answer" in source:
                     cell['source'] = [
                        "# Inline Question 2\n",
                        "\n",
                        "Despite their recent success in large-scale image recognition tasks, ViTs often lag behind traditional CNNs when trained on smaller datasets. What underlying factor contribute to this performance gap? What techniques can be used to improve the performance of ViTs on small datasets?\n",
                        "\n",
                        "**Your Answer**: \n",
                        "ViTs lack the inductive biases inherent in CNNs, such as translation invariance and locality. Consequently, they require much larger datasets to learn these properties from scratch. To improve performance on small datasets, we can use techniques like:\n",
                        "- **Strong Data Augmentation** (e.g., Mixup, CutMix)\n",
                        "- **Model Regularization** (e.g., Stochastic Depth, Weight Decay)\n",
                        "- **Knowledge Distillation** from a CNN teacher (as done in DeiT)\n",
                        "- **Pretraining** on larger datasets before fine-tuning.\n"
                     ]
                elif "Inline Question 3" in source and "Your Answer" in source:
                     cell['source'] = [
                        "# Inline Question 3\n",
                        "\n",
                        "How does the computational cost of the self-attention layers in a ViT change if we independently make the following changes? Please ignore the computation cost of QKV and output projection.\n",
                        "\n",
                        "(i) Double the hidden dimension.\n",
                        "(ii) Double the height and width of the input image.\n",
                        "(iii) Double the patch size.\n",
                        "(iv) Double the number of layers.\n",
                        "\n",
                        "**Your Answer**: \n",
                        "(i) **Linear**: Cost is $O(T^2 D)$. Doubling $D$ doubles the cost.\n",
                        "(ii) **Quadratic**: Doubling height and width increases number of patches $T$ by $4\\times$. Cost is $O(T^2 D)$, so it increases by $16\\times$.\n",
                        "(iii) **Inverse Quadratic**: Doubling patch size reduces number of patches $T$ by $4\\times$. Cost decreases by $16\\times$ (or becomes $1/16$ of original).\n",
                        "(iv) **Linear**: Doubling layers simply doubles the total cost.\n"
                     ]
            elif cell['cell_type'] == 'code':
                if "TODO: Train a Vision Transformer model" in source:
                    cell['source'] = [
                        "from icv83551.classification_solver_vit import ClassificationSolverViT\n",
                        "\n",
                        "############################################################################\n",
                        "# TODO: Train a Vision Transformer model that achieves over 0.45 test      #\n",
                        "# accuracy on CIFAR-10 after 2 epochs by adjusting the model architecture  #\n",
                        "# and/or training parameters as needed.                                    #\n",
                        "#                                                                          #\n",
                        "# Note: If you want to use a GPU runtime, go to `Runtime > Change runtime  #\n",
                        "# type` and set `Hardware accelerator` to `GPU`. This will reset Colab,    #\n",
                        "# so make sure to rerun the entire notebook from the beginning afterward.  #\n",
                        "############################################################################\n",
                        "\n",
                        "\n",
                        "learning_rate = 5e-4\n",
                        "weight_decay = 1e-2\n",
                        "batch_size = 128\n",
                        "model = VisionTransformer(dropout=0.1, num_layers=4, num_heads=4, embed_dim=256, dim_feedforward=512)\n",
                        "\n",
                        "\n",
                        "\n",
                        "\n",
                        "################################################################################\n",
                        "#                                 END OF YOUR CODE                             #\n",
                        "################################################################################\n",
                        "\n",
                        "solver = ClassificationSolverViT(\n",
                        "    train_data=train_data,\n",
                        "    test_data=test_data,\n",
                        "    model=model,\n",
                        "    num_epochs = 2,  # Don't change this\n",
                        "    learning_rate = learning_rate,\n",
                        "    weight_decay = weight_decay,\n",
                        "    batch_size = batch_size,\n",
                        ")\n",
                        "\n",
                        "solver.train('cuda' if torch.cuda.is_available() else 'cpu')\n",
                        "\n"
                    ]

        with open('Transformer_Captioning.ipynb', 'w', encoding='utf-8') as f:
            json.dump(nb, f, separators=(',', ':'))
        print("Updated Transformer_Captioning.ipynb")
    except Exception as e:
        print(f"Error updating Transformer_Captioning.ipynb: {e}")

if __name__ == "__main__":
    update_clip_dino()
    update_transformer_captioning()
    print("Notebooks updated successfully.")
