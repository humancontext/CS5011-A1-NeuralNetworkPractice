
import java.util.ArrayList;

import org.encog.ConsoleStatusReportable;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.pattern.FeedForwardPattern;
import org.encog.neural.prune.PruneIncremental;

public class Prunne {
	public static void main(String[] args){
		FeedForwardPattern net_test = new FeedForwardPattern();
		net_test.setInputNeurons(7);
		net_test.setOutputNeurons(5);
		net_test.setActivationFunction(new ActivationSigmoid());
		
		ArrayList<ArrayList<Double>> inputList = new ArrayList<ArrayList<Double>>();
		inputList = Learning1.initializeInput(inputList);
		double[][] INPUT = Learning1.readList(inputList);
		ArrayList<ArrayList<Double>> outputList = new ArrayList<ArrayList<Double>>();
		outputList = Learning1.initializeOutput(outputList);
		double[][] OUTPUT = Learning1.readList(outputList);
		MLDataSet trainingSet = new BasicMLDataSet(INPUT, OUTPUT);
		
		PruneIncremental prune= new PruneIncremental(trainingSet, net_test, 1000, 1, 10, new ConsoleStatusReportable());
		prune.addHiddenLayer(1, 5);
		prune.process();
		BasicNetwork network = prune.getBestNetwork();
		
		System.out.println("Neural Network created: "
		           + network.getLayerNeuronCount(0) +
		           "-" + network.getLayerNeuronCount(1) +
		        "-" + network.getLayerNeuronCount(2));
	}
}
