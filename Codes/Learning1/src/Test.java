import java.io.File;
import java.util.ArrayList;

import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.neural.networks.BasicNetwork;
import org.encog.persist.EncogDirectoryPersistence;

public class Test {
	public static void main(String[] args) {
		BasicNetwork loaded_net = (BasicNetwork) EncogDirectoryPersistence.loadObject(new File ("network_60.eg") ) ;
		ArrayList<ArrayList<Double>> inputList = new ArrayList<ArrayList<Double>>();
		inputList = initializeInput(inputList);
		double[][] INPUT = readList(inputList);
		ArrayList<ArrayList<Double>> outputList = new ArrayList<ArrayList<Double>>();
		outputList = initializeOutput(outputList);
		double[][] OUTPUT = readList(outputList);
		double sum = 0.0;
		for (int i = 0; i < 14; i++) {
			MLData data=new BasicMLData(INPUT[i]);
			MLData output = loaded_net.compute(data);
			boolean flag = true;
			double[] testOutput = new double[5];
			for (int j = 0; j < 5; j++) {
				testOutput[j] = 0.0;
				if(output.getData(j) >= 0.5) testOutput[j] = 1.0;
				if(testOutput[j] != OUTPUT[i][j]) flag = false;
			}
			
			if(flag) sum++;
		}
		System.out.println("The accuracy for classification is " + sum/14.0*100 + "%");
		
	}
	
	protected static ArrayList<ArrayList<Double>> initializeInput(ArrayList<ArrayList<Double>> input) {
		double[][] input_data = { 
				{0,1,0,1,0,0,0}, {0,0,0,0,1,0,0}, {0,1,1,0,0,0,1}, {1,0,0,0,0,1,0},
				{0,0,1,0,0,0,0}, {0,1,0,1,1,0,0}, {0,0,0,0,1,1,0}, {0,1,1,0,0,0,1},
				{1,0,0,0,0,1,1}, {0,0,1,0,0,1,0}, {0,1,0,0,1,0,0}, {0,0,0,1,1,1,0},
				{1,1,1,0,0,0,1}, {0,0,0,0,0,1,1}
				};
		input.addAll(writeList(input_data));
		return input;
	}
	
	protected static ArrayList<ArrayList<Double>> initializeOutput(ArrayList<ArrayList<Double>> output) {
		double[][] output_data = { 
				{1,0,0,0,0}, {0,1,0,0,0}, {0,0,1,0,0}, {0,0,0,1,0}, {0,0,0,0,1},
				{1,0,0,0,0}, {0,1,0,0,0}, {0,0,1,0,0}, {0,0,0,1,0}, {0,0,0,0,1},
				{1,0,0,0,0}, {0,1,0,0,0}, {0,0,1,0,0}, {0,0,0,1,0}
				};
		output.addAll(writeList(output_data));
		return output;
	}
	
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
}

