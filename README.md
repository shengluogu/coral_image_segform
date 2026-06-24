# coral_image_segform
Foundation Model Image Segmentation Competition Project

## 📈 Training Curves

<div align="center">
  <table>
    <tr>
      <td align="center"><b>mIOU Curve</b><br/><img src="assets/curves/miou_curve.png" width="400"/></td>
      <td align="center"><b>Class IOU Curve</b><br/><img src="assets/curves/class_iou_curve.png" width="400"/></td>
    </tr>
    <tr>
      <td align="center"><b>Accuracy Curve</b><br/><img src="assets/curves/accuracy_curve.png" width="400"/></td>
      <td align="center"><b>Loss Curve</b><br/><img src="assets/curves/loss_curve.png" width="400"/></td>
    </tr>
  </table>
</div>


## Qualitative Comparison

<div align="center">
    <p style="max-width: 800px; text-align: left; margin-bottom: 10px;">
    可观察到比赛提供的真实标签存在一定错标现象（如绿色圈出部分）。
    针对该问题，采用脏数据处理方法后，可有效降低噪声标签的影响。
  </p>
  <table style="border-collapse: collapse; border: none;">
    <tr>
      <!-- 第一行：普通场景 -->
      <td align="center" style="border: none; padding: 10px;">
        <b>Standard Prediction</b><br>
        <img src="assets/showcase/0435.png" width="350px" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/>
      </td>
      <td align="center" style="border: none; padding: 10px;">
        <b>Standard Prediction (Case 2)</b><br>
        <img src="assets/showcase/1185.png" width="350px" style="border-radius: 8px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);"/>
      </td>
    </tr>
    <tr>
      <!-- 第二行：脏数据处理场景（重点） -->
      <td align="center" style="border: none; padding: 10px; background-color: #f0f7ff;">
        <b>Dirty Data Handling ✨</b><br>
        <img src="assets/showcase/0237.png" width="350px" style="border: 2px solid #0969da; border-radius: 8px;"/>
      </td>
      <td align="center" style="border: none; padding: 10px; background-color: #f0f7ff;">
        <b>Dirty Data Handling ✨</b><br>
        <img src="assets/showcase_dity/0995.png" width="350px" style="border: 2px solid #0969da; border-radius: 8px;"/>
      </td>
    </tr>
  </table>
</div>
## 
