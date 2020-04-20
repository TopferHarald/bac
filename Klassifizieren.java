import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import org.tensorflow.Graph;
import org.tensorflow.Output;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class Klassifizieren {

    public static void main(String[] args) {
        String imagePath = args[0];
        String modelPath = args[1];

        byte[] graphDef = null;
        byte[] imageBytes = null;
        List<String> labels = null;

        try {
            graphDef = Files.readAllBytes(Paths.get(modelPath, "retrained_graph.pb"));
            imageBytes = Files.readAllBytes(Paths.get(imagePath));
            labels = Files.readAllLines(Paths.get(modelPath, "retrained_labels.txt"), Charset.forName("UTF-8"));
        } catch (IOException e) {
            System.err.println("Fehler beim Lesen!");
            System.exit(1);
        }

        try (Tensor<Float> image = createGraph(imageBytes)) {
            float[] labelProbabilities = calculateProbPerCategory(graphDef, image);
            int bestLabelIdx = getKategorie(labelProbabilities);
            System.out.println(
                    String.format("Kategorie: %s (mit Wahrscheinlichkeit von %.2f%%)",
                            labels.get(bestLabelIdx),
                            labelProbabilities[bestLabelIdx] * 100));
        }
    }

    private static Tensor<Float> createGraph(byte[] image) {
        try (Graph g = new Graph()) {
            GraphUtil gu = new GraphUtil(g);

            //Hier werden die Größenangaben der Bilder erwartet die für das Lernen verwendet wurden
            int Height = 299;
            int Width = 299;
            float mean = 117;
            float scale = 1;

            final Output input = gu.constant("input", image);
            final Output output =
                    gu.div(gu.sub(gu.resizeBilinear(gu.expandDims(
                            gu.cast(gu.decode(input, 3), Float.class),
                            gu.constant("make_batch", 0)),
                            gu.constant("size", new int[] {Height, Width})),
                            gu.constant("mean", mean)),
                            gu.constant("scale", scale));
            try (Session s = new Session(g)) {
                return s.runner().fetch(output.op().name()).run().get(0).expect(Float.class);
            }
        }
    }

    private static float[] calculateProbPerCategory(byte[] graph, Tensor<Float> image) {
        try (Graph g = new Graph()) {
            g.importGraphDef(graph);
            try (Session s = new Session(g);
                 Tensor<Float> result = s.runner().feed("Mul:0", image).fetch("final_result:0").run().get(0).expect(Float.class)) {
                    final long[] shapeResult = result.shape();

                    if (result.numDimensions() != 2 || shapeResult[0] != 1)
                        throw new RuntimeException();

                    int numberLabels = (int) shapeResult[1];
                    return result.copyTo(new float[1][numberLabels])[0];
            }
        }
    }

    private static int getKategorie(float[] probs) {
        int result = 0;
        for (int i = 1; i < probs.length; ++i) {
            if (probs[i] > probs[result]) {
                result = i;
            }
        }
        return result;
    }
}