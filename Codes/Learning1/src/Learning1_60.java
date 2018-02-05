/**
 * Part 1
 * @author XignzhiYue CS5011 Student (xy31@st-andrews.ac.uk)
 * @version 1.0
 * @since 2017-10-07
 */

import java.io.*;
import java.util.*;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.persist.EncogDirectoryPersistence;

public class Learning1_60 {
	public static void main(String[] args) throws Exception {
		//Set the input units
		int input_units = 7;
		//Set the units in hidden layer
		int hidden_units = 3;
		//Set the output units
		int output_units = 5;
		
		BasicNetwork network = new BasicNetwork();
		
		//Create the Input Layer
		network.addLayer(new BasicLayer(null, false, input_units));
		
		//Hidden Layer
		network.addLayer(new BasicLayer(new ActivationSigmoid(), true, hidden_units));
		
		//Output Layer
		network.addLayer(new BasicLayer(new ActivationSigmoid(), false, output_units));
		
		network.getStructure().finalizeStructure();
		network.reset();
		
		//Setting up the network with inputs and outputs
		System.out.println("Network created with structure " + input_units + "-" + hidden_units + "-" + output_units);
		ArrayList<ArrayList<Double>> inputList = new ArrayList<ArrayList<Double>>();
		inputList = initializeInput(inputList);
		double[][] INPUT = readList(inputList);
		ArrayList<ArrayList<Double>> outputList = new ArrayList<ArrayList<Double>>();
		outputList = initializeOutput(outputList);
		double[][] OUTPUT = readList(outputList);
 		MLDataSet trainingSet = new BasicMLDataSet(INPUT, OUTPUT);
		
		//training with learning rate=0.3 and momentum=0.3
		Backpropagation train=new Backpropagation (network, trainingSet,0.3,0.3);
		int epoch = 1;
		
		do {
			train.iteration();
			System.out.println("Epoch #" + epoch + " Error:" + train.getError());
			epoch++;
		   } while(train.getError() > 0.01);
		train.finishTraining();

		//testing the network with a set sample 
		double[] h=new double[]{0,1,0,1,0,0,0};
		System.out.println("Test input:");
		System.out.println("no "+"yes "+"no "+"yes "+"no "+"no "+"no");
		MLData data=new BasicMLData(h);
		MLData output = network.compute(data);
		System.out.println("actual = ");
		for(int i = 0; i < 5; i++) {
			System.out.println(" " + output.getData(i));
		}

		//saving the network
		System.out.println("Saving network");
		EncogDirectoryPersistence.saveObject(new File("network_60.eg"), network);
	}

	/** Transfer data from a 2D array list to a 2D array
	*@param list is the array list
	*@return double[][] a 2D array of doubles that stores the data.
	*/
	protected static double[][] readList(ArrayList<ArrayList<Double>> list){
        //transfer the data into the 2D array.
        double[][] data = new double[list.size()][list.get(0).size()];
        for(int i = 0; i < list.size(); i++){
            for(int j = 0; j < list.get(0).size(); j++){
                data[i][j] = list.get(i).get(j);
            }
        }
        return data;
	}
	
	/** Transfer data from a 2D array to a 2D array list
	*@param list is the 2D array of double
	*@return the 2D array list that we can add some more data to it
	*/
	protected static ArrayList<ArrayList<Double>> writeList(double[][] data){
        //transfer the 2D array into a arrayList.
		ArrayList<ArrayList<Double>> list = new ArrayList<ArrayList<Double>>();
		for(double[] i: data) {
			ArrayList<Double> temp = new ArrayList<Double>();
			for(double j: i) {
				temp.add(j);
			}
			list.add(temp);
		}
		return list;
	}
	
	/**
	* this method is used to initialize the input database.
	* @param 	a blank 2D array list
	* @return	input.
	*/
	protected static ArrayList<ArrayList<Double>> initializeInput(ArrayList<ArrayList<Double>> input) {
		double[][] input_data = { 
				{0,1,0,1,0,0,0}, {0,0,0,0,1,0,0}, {0,1,1,0,0,0,1}, {1,0,0,0,0,1,0},
				{0,0,1,0,0,0,0}, {0,1,0,1,1,0,0}, {0,0,0,0,1,1,0}, {0,1,1,0,0,0,1},
				{1,0,0,0,0,1,1}, {0,0,1,0,0,1,0}, {0,1,0,0,1,0,0}, {0,0,0,1,1,1,0},
				{1,1,1,0,0,0,1}, {0,0,0,0,0,1,1}, {0,0,1,0,0,1,0}, {0,0,0,1,1,0,0},
				{0,0,0,1,1,1,0}, {1,1,0,0,0,0,1}, {1,0,0,0,0,1,1}, {0,0,1,0,0,1,0},
				{0,1,0,0,1,0,0}, {0,0,0,1,1,1,0}, {1,1,1,0,0,0,1}, {1,0,0,0,0,1,1},
				{0,0,1,0,0,1,0}, {0,1,0,1,0,0,0}, {0,0,0,0,1,0,0}, {0,1,1,0,0,0,1},
				{1,0,0,0,0,1,0}, {0,0,1,0,0,0,0}, {0,1,0,1,1,0,0}, {0,0,0,0,1,1,0},
				{0,1,1,0,0,0,1}, {1,0,0,0,0,1,1}, {0,0,1,0,0,1,0}
				};
		input.addAll(writeList(input_data));
		return input;
	}
	
	/**
	* this method is used to initialize the output database.
	* @param 	a blank 2D array list
	* @return	output.
	*/
	protected static ArrayList<ArrayList<Double>> initializeOutput(ArrayList<ArrayList<Double>> output) {
		double[][] output_data = { 
				{1,0,0,0,0}, {0,1,0,0,0}, {0,0,1,0,0}, {0,0,0,1,0}, {0,0,0,0,1},
				{1,0,0,0,0}, {0,1,0,0,0}, {0,0,1,0,0}, {0,0,0,1,0}, {0,0,0,0,1},
				{1,0,0,0,0}, {0,1,0,0,0}, {0,0,1,0,0}, {0,0,0,1,0}, {0,0,0,0,1},
				{1,0,0,0,0}, {0,1,0,0,0}, {0,0,1,0,0}, {0,0,0,1,0}, {0,0,0,0,1},
				{1,0,0,0,0}, {0,1,0,0,0}, {0,0,1,0,0}, {0,0,0,1,0}, {0,0,0,0,1},
				{1,0,0,0,0}, {0,1,0,0,0}, {0,0,1,0,0}, {0,0,0,1,0}, {0,0,0,0,1},
				{1,0,0,0,0}, {0,1,0,0,0}, {0,0,1,0,0}, {0,0,0,1,0}, {0,0,0,0,1}
				};
		output.addAll(writeList(output_data));
		return output;
	}

}
