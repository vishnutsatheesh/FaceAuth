import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_face_detection/google_mlkit_face_detection.dart';
import 'package:image/image.dart' as img_lib;
import 'package:tflite_flutter/tflite_flutter.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  final firstCamera = cameras.first;
  runApp(MyApp(camera: firstCamera));
}

class MyApp extends StatelessWidget {
  final CameraDescription camera;

  const MyApp({required this.camera, super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Face Auth Demo',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: FaceAuthPage(camera: camera),
    );
  }
}

class FaceAuthPage extends StatefulWidget {
  final CameraDescription camera;

  const FaceAuthPage({required this.camera, super.key});

  @override
  State<FaceAuthPage> createState() => _FaceAuthPageState();
}

class _FaceAuthPageState extends State<FaceAuthPage> {
  late CameraController _controller;
  bool _isDetecting = false;
  final FaceDetector _faceDetector = FaceDetector(
    options: FaceDetectorOptions(
      enableContours: false,
      enableClassification: false,
      performanceMode: FaceDetectorMode.fast,
    ),
  );

  Interpreter? _interpreter;
  List<double>? _storedEmbedding;

  @override
  void initState() {
    super.initState();
    _initCamera();
    _loadModel();
    _prepareStoredEmbedding();
  }

  @override
  void dispose() {
    _controller.dispose();
    _faceDetector.close();
    _interpreter?.close();
    super.dispose();
  }

  Future<void> _initCamera() async {
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.medium,
      enableAudio: false,
    );
    await _controller.initialize();
    await _controller.startImageStream(_processCameraImage);
    setState(() {});
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('facenet.tflite');
      print('Interpreter loaded');
    } catch (e) {
      print('Failed to load model: $e');
    }
  }

  Future<void> _prepareStoredEmbedding() async {
    final byteData = await rootBundle.load('assets/prestored_face.jpg');
    final bytes = byteData.buffer.asUint8List();
    final embedding = await _imageToEmbedding(bytes);
    setState(() {
      _storedEmbedding = embedding;
    });
    print('Stored embedding prepared');
  }

  void _processCameraImage(CameraImage image) async {
    if (_isDetecting) return;
    _isDetecting = true;
    try {
      final allBytes = WriteBuffer();
      for (final plane in image.planes) {
        allBytes.putUint8List(plane.bytes);
      }
      final bytes = allBytes.done().buffer.asUint8List();

      final Size imageSize = Size(
        image.width.toDouble(),
        image.height.toDouble(),
      );
      final InputImageRotation imageRotation = InputImageRotation.rotation0deg;
      final InputImageFormat inputImageFormat =
          InputImageFormatMethods.fromRawValue(image.format.raw) ??
          InputImageFormat.NV21;
      final planeData = image.planes
          .map(
            (plane) => InputImagePlaneMetadata(
              bytesPerRow: plane.bytesPerRow,
              height: plane.height,
              width: plane.width,
            ),
          )
          .toList();

      final im = InputImage.fromBytes(
        bytes: bytes,
        inputImageData: InputImageData(
          size: imageSize,
          imageRotation: imageRotation,
          inputImageFormat: inputImageFormat,
          planeData: planeData,
        ),
      );

      final faces = await _faceDetector.processImage(im);
      if (faces.isNotEmpty) {
        final face = faces.first;
        final png = await _convertYUV420toImage(image);
        final rect = Rect.fromLTRB(
          max(0, face.boundingBox.left),
          max(0, face.boundingBox.top),
          min(png.width.toDouble(), face.boundingBox.right),
          min(png.height.toDouble(), face.boundingBox.bottom),
        );
        final crop = img_lib.copyCrop(
          png,
          rect.left.toInt(),
          rect.top.toInt(),
          rect.width.toInt(),
          rect.height.toInt(),
        );
        final jpg = img_lib.encodeJpg(crop);
        final emb = await _imageToEmbedding(Uint8List.fromList(jpg));

        if (_storedEmbedding != null && emb != null) {
          final dist = _cosineDistance(_storedEmbedding!, emb);
          print('distance: $dist');
          if (dist < 0.4) {
            print('AUTHORIZED');
            _showAuthorized();
          } else {
            print('NOT AUTHORIZED');
          }
        }
      }
    } catch (e) {
      print('Error processing camera image: $e');
    } finally {
      _isDetecting = false;
    }
  }

  Future<img_lib.Image> _convertYUV420toImage(CameraImage image) async {
    try {
      final XFile file = await _controller.takePicture();
      final bytes = await file.readAsBytes();
      final decoded = img_lib.decodeImage(bytes)!;
      return decoded;
    } catch (e) {
      rethrow;
    }
  }

  Future<List<double>?> _imageToEmbedding(Uint8List imageBytes) async {
    if (_interpreter == null) return null;
    final image = img_lib.decodeImage(imageBytes)!;
    final resized = img_lib.copyResize(image, width: 160, height: 160);
    final input = Float32List(1 * 160 * 160 * 3);
    int idx = 0;
    for (int y = 0; y < 160; y++) {
      for (int x = 0; x < 160; x++) {
        final pixel = resized.getPixel(x, y);
        final r = img_lib.getRed(pixel);
        final g = img_lib.getGreen(pixel);
        final b = img_lib.getBlue(pixel);
        input[idx++] = (r - 128) / 128;
        input[idx++] = (g - 128) / 128;
        input[idx++] = (b - 128) / 128;
      }
    }

    final output = List.filled(192, 0.0).reshape([1, 192]);
    try {
      _interpreter!.run(input.reshape([1, 160, 160, 3]), output);
    } catch (e) {
      print('Interpreter run error: $e');
      return null;
    }

    final emb = List<double>.from(output[0]);
    final norm = sqrt(emb.map((e) => e * e).reduce((a, b) => a + b));
    return emb.map((e) => e / norm).toList();
  }

  double _cosineDistance(List<double> a, List<double> b) {
    double dot = 0.0;
    double na = 0.0;
    double nb = 0.0;
    for (int i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    final denom = sqrt(na) * sqrt(nb);
    if (denom == 0) return 1.0;
    final cos = dot / denom;
    return 1 - cos;
  }

  void _showAuthorized() {
    if (!mounted) return;
    ScaffoldMessenger.of(
      context,
    ).showSnackBar(const SnackBar(content: Text('AUTHORIZED ✅')));
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Face Auth Demo')),
      body: _controller.value.isInitialized
          ? CameraPreview(_controller)
          : const Center(child: CircularProgressIndicator()),
    );
  }
}

extension ListReshape on List {
  List reshape(List<int> dims) => this;
}
