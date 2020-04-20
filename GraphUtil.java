import org.tensorflow.DataType;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Tensor;

/**
 * Util-Klasse um den Graph zu erzeugen und diverse Operationen darauf anwenden zu können.
 */
public class GraphUtil {
    private Graph graph;

    public GraphUtil(Graph graph) {
        this.graph = graph;
    }

    public Output div(Output x, Output y) {
        return binaryOp("Div", x, y);
    }

    public Output sub(Output x, Output y) {
        return binaryOp("Sub", x, y);
    }

    public Output resizeBilinear(Output images, Output size) {
        return binaryOp3("ResizeBilinear", images, size);
    }

    public Output expandDims(Output input, Output dim) {
        return binaryOp3("ExpandDims", input, dim);
    }

    public Output cast(Output value, Class type) {
        DataType dtype = DataType.fromClass(type);
        return graph.opBuilder("Cast", "Cast")
                .addInput(value)
                .setAttr("DstT", dtype)
                .build()
                .output(0);
    }

    public Output decode(Output contents, long channels) {
        return graph.opBuilder("DecodeJpeg", "DecodeJpeg")
                .addInput(contents)
                .setAttr("channels", channels)
                .build()
                .output(0);
    }

    public Output constant(String name, Object value, Class type) {
        try (Tensor t = Tensor.create(value, type)) {
            return graph.opBuilder("Const", name)
                    .setAttr("dtype", DataType.fromClass(type))
                    .setAttr("value", t)
                    .build()
                    .output(0);
        }
    }

    public Output<String> constant(String name, byte[] value) {
        return this.constant(name, value, String.class);
    }

    public Output<Integer> constant(String name, int value) {
        return this.constant(name, value, Integer.class);
    }

    public Output<Integer> constant(String name, int[] value) {
        return this.constant(name, value, Integer.class);
    }

    public Output<Float> constant(String name, float value) {
        return this.constant(name, value, Float.class);
    }

    private Output binaryOp(String type, Output out1, Output out2) {
        return graph.opBuilder(type, type).addInput(out1).addInput(out2).build().output(0);
    }

    private Output binaryOp3(String type, Output out1, Output out2) {
        return graph.opBuilder(type, type).addInput(out1).addInput(out2).build().output(0);
    }
}