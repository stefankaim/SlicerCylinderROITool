import slicer
import vtk
import qt
import math
import vtkSegmentationCore
import numpy as np
import csv
import os
import uuid

class CylinderROIPanel:
    def __init__(self):
        self.segmentationNode = None
        self.widget = qt.QWidget()
        self.widget.setWindowTitle("Cylindrical ROI Tool + Statistics Export")
        self.layout = qt.QFormLayout(self.widget)

        self.diameterBox = qt.QDoubleSpinBox()
        self.diameterBox.setSuffix(" mm")
        self.diameterBox.setDecimals(3)
        self.diameterBox.setValue(20.0)
        self.layout.addRow("Diameter:", self.diameterBox)

        self.heightBox = qt.QDoubleSpinBox()
        self.heightBox.setSuffix(" mm")
        self.heightBox.setDecimals(3)
        self.heightBox.setValue(30.0)
        self.layout.addRow("Height:", self.heightBox)
        
        self.cylinderButton = qt.QPushButton("Only generate Cylinders")
        self.cylinderButton.clicked.connect(self.onlyCylinders)
        self.layout.addRow(self.cylinderButton)

        self.directoryButton = qt.QPushButton("Choose Export-Folder")
        self.directoryButton.clicked.connect(self.selectDirectory)
        self.layout.addRow("Export-Folder:", self.directoryButton)
        self.outputDirectory = slicer.app.temporaryPath


        self.statisticButton = qt.QPushButton("Only export statistics")
        self.statisticButton.clicked.connect(self.onlyStatistic)
        self.layout.addRow(self.statisticButton)

        self.bothButton = qt.QPushButton("Generate Cylinders + Export")
        self.bothButton.clicked.connect(self.generateAndExport)
        self.layout.addRow(self.bothButton)

        self.widget.setLayout(self.layout)
        self.widget.show()

    def selectDirectory(self):
        dir = qt.QFileDialog.getExistingDirectory()
        if dir:
            self.outputDirectory = dir
            self.directoryButton.setText(dir)

    def generateCylinder(self, diameter, height):
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", "Cylinder_Segments")
        segmentationNode.CreateDefaultDisplayNodes()
        
        referenceVolume = slicer.util.getNode('vtkMRMLScalarVolumeNode*')
        
        
        image = referenceVolume.GetImageData()
        matrix = vtk.vtkMatrix4x4()
        referenceVolume.GetRASToIJKMatrix(matrix)
        
        spacing = referenceVolume.GetSpacing()
        origin = referenceVolume.GetOrigin()
        
        elements = [
            f"{spacing[0]}", f"{spacing[1]}", f"{spacing[2]}",
            f"{origin[0]}", f"{origin[1]}", f"{origin[2]}"
        ]
        for row in range(3):
            for col in range(3):
                elements.append(str(matrix.GetElement(row, col)))

        geometryString = ";".join(elements)
        segmentationNode.SetAttribute("referenceImageGeometryRef", geometryString)

        allNodes = slicer.util.getNodesByClass("vtkMRMLMarkupsFiducialNode")
        for punktNode in allNodes:
            if punktNode.GetNumberOfControlPoints() != 1:
                continue

            coords = [0.0, 0.0, 0.0]
            pointNode.GetNthControlPointPosition(0, coords)
            pointName = punktNode.GetName()
            segmentName = f"Cylinder_{pointName}"

            radius = diameter / 2.0

            # Anzahl Voxel berechnen
            extentX = int(radius / spacing[0])
            extentY = int(radius / spacing[1])
            extentZ = int(height / 2.0 / spacing[2])

            imageData = vtk.vtkImageData()
            imageData.SetSpacing(spacing)
            imageData.SetDimensions(extentX * 2, extentY * 2, extentZ * 2)
            imageData.SetOrigin(coords[0] - extentX * spacing[0], coords[1] - extentY * spacing[1], coords[2] - extentZ * spacing[2])
            imageData.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

            for z in range(-extentZ, extentZ):
                for y in range(-extentY, extentY):
                    for x in range(-extentX, extentX):
                        wx = x * spacing[0]
                        wy = y * spacing[1]
                        wz = z * spacing[2]
                        r2 = wx ** 2 + wy ** 2
                        if r2 <= radius ** 2:
                            imageData.SetScalarComponentFromDouble(x + extentX, y + extentY, z + extentZ, 0, 1)
            
            labelmapNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            labelmapNode.SetAndObserveImageData(imageData)
            labelmapNode.SetName(segmentName)
            labelmapNode.SetSpacing(spacing)
            labelmapNode.SetOrigin(imageData.GetOrigin())

            slicer.vtkSlicerSegmentationsModuleLogic().ImportLabelmapToSegmentationNode(labelmapNode, segmentationNode)

            slicer.mrmlScene.RemoveNode(labelmapNode)

        return segmentationNode

    def exportStatistics(self, segmentationNode, volumeNode):
        import vtk.util.numpy_support as ns

        imageData = volumeNode.GetImageData()
        labelMap = vtk.vtkImageData()
        spacing = volumeNode.GetSpacing()
        origin = volumeNode.GetOrigin()
        dims = imageData.GetDimensions()

        segmentIDs = vtk.vtkStringArray()
        segmentationNode.GetSegmentation().GetSegmentIDs(segmentIDs)

        for i in range(segmentIDs.GetNumberOfValues()):
            segmentID = segmentIDs.GetValue(i)
            segmentName = segmentationNode.GetSegmentation().GetSegment(segmentID).GetName()
            if not segmentName.startswith("Cylinder_"):
                continue

            labelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            singleID = vtk.vtkStringArray()
            singleID.InsertNextValue(segmentID)
            slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(segmentationNode, singleID, labelNode, volumeNode)
            labelImage = labelNode.GetImageData()

            img_array = ns.vtk_to_numpy(imageData.GetPointData().GetScalars()).reshape(dims[::-1])
            label_array = ns.vtk_to_numpy(labelImage.GetPointData().GetScalars()).reshape(dims[::-1])

            exportPath = os.path.join(self.outputDirectory, f"Statistic_{segmentName}.csv")
            rows = ["Slice_Z_mm;Mean;StdDev;Min;Max;StdError"]

            for z in range(dims[2]):
                mask = label_array[z] > 0
                values = img_array[z][mask]
                if values.size > 0:
                    mean = np.mean(values)
                    std = np.std(values)
                    minv = np.min(values)
                    maxv = np.max(values)
                    stderr = std / np.sqrt(values.size)
                    z_mm = origin[2] + z * spacing[2]
                    rows.append(f"{z_mm:.2f};{mean:.2f};{std:.2f};{minv:.2f};{maxv:.2f};{stderr:.2f}".replace('.', ','))

            with open(exportPath, 'w', encoding='utf-8') as f:
                f.write('\n'.join(rows))

            slicer.mrmlScene.RemoveNode(labelNode)

    def onlyCylinders(self):
        diameter = self.diameterBox.value
        height = self.heightBox.value
        self.segmentationNode = self.generateCylinder(diameter, height)
        slicer.util.infoDisplay("Cylinder generated.")

    def onlyStatistic(self):
        volumeNode = slicer.util.getNode('vtkMRMLScalarVolumeNode*')
        if not volumeNode or self.segmentationNode is None:
            slicer.util.errorDisplay("Cylinder or scan not found.")
            return
        self.exportStatistics(self.segmentationNode, volumeNode)
        slicer.util.infoDisplay("Statistic exported.")

    def generateAndExport(self):
        diameter = self.diameterBox.value
        height = self.heightBox.value

        volumeNode = slicer.util.getNode('vtkMRMLScalarVolumeNode*')
        if not volumeNode:
            slicer.util.errorDisplay("No volume found.")
            return

        self.segmentationNode = self.generateCylinder(diameter, height)
        self.exportStatistics(self.segmentationNode, volumeNode)
        slicer.util.infoDisplay(f"Generated cylinders and exported statistics.")

cylinderTool = CylinderROIPanel()
