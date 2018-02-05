/**
 * Part 2
 * @author XignzhiYue CS5011 Student (xy31@st-andrews.ac.uk)
 * @version 1.0
 * @since 2017-10-07
 */
import java.io.File;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Scanner;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.persist.EncogDirectoryPersistence;

public class Learning2_addChara {
	
	public static void main(String[] args) {
		//initializing the quetion and name maps
		ArrayList<String> nameList = new ArrayList<String>(){{add("Alex"); add("Alfred"); add("Anita"); add("Anne"); add("Bernard");}};
		//add a new character Thomas that gives all positive answers to the questions.
		nameList.add("Thomas");
		
		ArrayList<String> questionList = new ArrayList<String>() {
			{	add("Does your character have curly hair?"); 
				add("Is your character blonde?"); 
				add("Does your character have red cheek?");
				add("Does your character have moustache?"); 
				add("Does your character have beard?"); 
				add("Does your character wear ear rings?");
				add("Is your character female?");
			}
		};
		//loading the network
		System.out.println("Loading network");
		Learning2_addChara l2 = new Learning2_addChara();
		
		ArrayList<ArrayList<Double>> inputList = new ArrayList<ArrayList<Double>>();
		inputList = initializeInput(inputList);
		ArrayList<ArrayList<Double>> outputList = new ArrayList<ArrayList<Double>>();
		outputList = initializeOutput(outputList);
		l2.play(nameList, questionList, inputList, outputList);
	}
		
	
	public void play(ArrayList<String> nameList, ArrayList<String> questionList, ArrayList<ArrayList<Double>> inputList, ArrayList<ArrayList<Double>> outputList) {
		//initializing the game
		System.out.println("Loading network");
		BasicNetwork loaded_net = this.Learning(inputList, outputList, 3);
		double[] INPUT=new double[questionList.size()];
		ArrayList<Integer> wrongGuess = new ArrayList<Integer>();
		boolean flag = false;

		//start playing
		System.out.println("Let's play the guessWho game! \nPlease answer yes/no for each question");
		
		for (int i = 0; i < questionList.size(); i++) {
			INPUT[i] = Judge(questionList.get(i));
			/** Early guessing
			*	input1 assumes the answers for the rest questions are y
			*	input2 assumes the answers for the rest questions are n
			*	if the guessing for the two inputs are identical and the guess wasn't on the wrong guess list
			*	make early guess
			*/
			double[] guessINPUT = new double[questionList.size()];
			for (int j = 0; j <= i; j++) {
				guessINPUT[j] = INPUT[j];
			}
			for (int j = i + 1; j < questionList.size(); j++) {
				guessINPUT[j] = 1.0;
			}
			MLData data1=new BasicMLData(INPUT);
			MLData output1 = loaded_net.compute(data1);
			MLData data2=new BasicMLData(guessINPUT);
			MLData output2 = loaded_net.compute(data2);
			
			//an extra condition >0.5 as threshold: see method
			if(Find(output1, nameList.size()) == Find(output2, nameList.size()) && Find(output1, nameList.size()) != -1 && Find(output2, nameList.size()) != -1 && !wrongGuess.contains(Find(output1, nameList.size()))) {
				System.out.println("I am guessing " + nameList.get(Find(output1, nameList.size())));
				System.out.println("Am I right? yes/no");
				Scanner in = new Scanner(System.in);
			    String s = in.nextLine();
			    while( !s.equals("no") && !s.equals("yes")) {
	    	    	System.out.println("Please answer \"yes\" or \"no\".");
	    	    	in = new Scanner(System.in);
	    		    s = in.nextLine();
	    	    }
			    if(s.equals("yes")) {
			    	flag = true;
			    	break;
			    }
			    else {
			    	wrongGuess.add(Find(output1, nameList.size()));
			    }
			}
		}
		if(flag) {
			System.out.println("Great! Wanna try again?");
		}
		else {
			//in case all early guessing does not work, guess the names one by one with greatest value for one-hot coding
			MLData data=new BasicMLData(INPUT);
			MLData output = loaded_net.compute(data);
			//rank by the output value for one-hot coding
			int[] rank = Rank(output, nameList.size());
			int index = 0;
			String s = "no";
			do{
				while(wrongGuess.contains(rank[index])) {
					index++;
				}
				System.out.println("My guess will be " + nameList.get(rank[index]));
				System.out.println("Am I right? yes/no");
				Scanner in = new Scanner(System.in);
			    s = in.nextLine();
			    while( !s.equals("no") && !s.equals("yes")) {
	    	    	System.out.println("Please answer \"yes\" or \"no\".");
	    	    	in = new Scanner(System.in);
	    		    s = in.nextLine();
	    	    }
			    index++;
			} while(s.equals("no") && index<nameList.size());
			if(index > nameList.size()) {
				System.out.println("Sorry this character is not recorded.");
			}
			else System.out.println("Great! Wanna try again?");
		}
	}
	/**
	* this method is used to ask questions and generate input data from valid answers.
	* @param 	question is the question to be asked.
	* @return	double is the input data concerning the given question.
	*/
	private static double Judge(String question) {
		double ans = 0;
		System.out.println(question);
		Scanner in = new Scanner(System.in);
	    String s = in.nextLine();
	    while( !s.equals("no") && !s.equals("yes")) {
	    	System.out.println("Please answer yes or no for each question.");
	    	in = new Scanner(System.in);
		    s = in.nextLine();
	    }
		if (s.equals("yes"))  ans = 1;
		return ans;
	}
	
	/**
	* this method is used to set the threshold, i.e. to find the biggest output 
	* factor (must >0.5) and set it to be 1, and others to be 0.
	* @param 	output is calculated with given input from the network.
	* @return	int is the factor that is set to be 1 (-1 if all is <0.5).
	*/
	private static int Find(MLData output, int num) {
		double val = 0.5;
		int index = -1;
		for (int i = 0; i < num; i++) {
			if(output.getData(i) > val) {
				index = i;
				val = output.getData(i);
			}
		}
		return index;
	}
	
	/**
	* this method is used to rank the output by its value for different factors.
	* @param 	output is calculated with given input from the network.
	* @return	int[] is ranking for all the output units.
	*/
	private static int[] Rank(MLData output, int num) {
		double array[] = new double[num];
		int rank[] = new int[num];
		for (int i = 0; i < num; i++) {
			array[i] = output.getData(i);
			rank[i] = i;
		}
		for (int i = 0; i < num-1; i++) {
			for (int j = i + 1; j < num; j++) {
				if(array[i] < array[j]) {
					double temp1 = array[i];
					array[i] = array[j];
					array[j] = temp1;
					int temp2 = rank[i];
					rank[i] = rank[j];
					rank[j] = temp2;
				}
			}
		}
		return rank;
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
				{0,1,1,0,0,0,1}, {1,0,0,0,0,1,1}, {0,0,1,0,0,1,0}, {0,0,0,0,0,0,0}//this is the features for Thomas
//				{0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}, {0,0,0,0,0,0,0}
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
				{1,0,0,0,0,0}, {0,1,0,0,0,0}, {0,0,1,0,0,0}, {0,0,0,1,0,0}, {0,0,0,0,1,0},
				{1,0,0,0,0,0}, {0,1,0,0,0,0}, {0,0,1,0,0,0}, {0,0,0,1,0,0}, {0,0,0,0,1,0},
				{1,0,0,0,0,0}, {0,1,0,0,0,0}, {0,0,1,0,0,0}, {0,0,0,1,0,0}, {0,0,0,0,1,0},
				{1,0,0,0,0,0}, {0,1,0,0,0,0}, {0,0,1,0,0,0}, {0,0,0,1,0,0}, {0,0,0,0,1,0},
				{1,0,0,0,0,0}, {0,1,0,0,0,0}, {0,0,1,0,0,0}, {0,0,0,1,0,0}, {0,0,0,0,1,0},
				{1,0,0,0,0,0}, {0,1,0,0,0,0}, {0,0,1,0,0,0}, {0,0,0,1,0,0}, {0,0,0,0,1,0},
				{1,0,0,0,0,0}, {0,1,0,0,0,0}, {0,0,1,0,0,0}, {0,0,0,1,0,0}, {0,0,0,0,1,0},
				{0,0,0,0,0,1}//Thomas lives here
//				, {0,0,0,0,0,1}, {0,0,0,0,0,1}, {0,0,0,0,0,1}, {0,0,0,0,0,1} 

				};
		output.addAll(writeList(output_data));
		return output;
	}
	
	/**
	* this method is used learn the given input-output pairs - basically what's done in part 1
	* @param 	input_data
	* @param	output_data
	* @return	BasicNetwork to be used to play the game
	*/
	protected BasicNetwork Learning(ArrayList<ArrayList<Double>> input_data, ArrayList<ArrayList<Double>> output_data, int num) {
		//Set the input units
		int input_units = input_data.get(0).size();
		//Set the units in hidden layer
		int hidden_units = num;
		//Set the output units
		int output_units = output_data.get(0).size();
		System.out.println("Training with new data.");
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

		double[][] INPUT = readList(input_data);
		double[][] OUTPUT = readList(output_data);
 		MLDataSet trainingSet = new BasicMLDataSet(INPUT, OUTPUT);
		
		//training with learning rate=0.1 and momentum=0
		Backpropagation train=new Backpropagation (network, trainingSet,0.3,0.3);
//		int epoch = 1;
		do {
			train.iteration();
//			System.out.println(train.getError());
//			epoch++;
		   } while(train.getError() > 0.01);
		train.finishTraining();
		train.finishTraining();
		return network;
	}
	
	/**
	* this method is used to update the input database.
	* @param 	maybe records which questions that the player answered maybe
	* @param	inputList is the original input database.
	* @param	INPUT records the players answers ("maybe"s are treated as "no"s here)
	* @return	updated input database.
	*/
	protected static ArrayList<ArrayList<Double>> addInput(ArrayList<ArrayList<Double>> inputList, double[] INPUT) {
		ArrayList<Double> temp = new ArrayList<Double>();
		
		for(int i = 0; i < INPUT.length; i++) {
			temp.add(INPUT[i]);
		}
		
		inputList.add(temp);
		return inputList;
	}
}


