**Run**

```bash
$ python [newmain.py](http://newmain.py/)
```

1. Click Register for inputting new face
2. Input name
3. Click enter
4. The generated embeddings will be stored as JSON array in facestored.txt, an image wont be saved. You can add cv2.imwrite if required.

Our use case required only taking one full frontal frame, If needed for higher accuracies you can take average/concatenation of multiple embeddings generated, add side view for faces.

You can train SVM classifier on this JSON data for further better recognition accuracy instead of solely depending on distance metrics. Will be adding it in later versions

 5. Click verify for Recognition

If face data is already present it will output bounding boxes with label. If not, will throw dialog box suggesting to register again(after 25 frames).
