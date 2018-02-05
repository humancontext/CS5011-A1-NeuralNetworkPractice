/**
 * Part 3 added "maybe" situation
 * 
 * @author XignzhiYue CS5011 Student (xy31@st-andrews.ac.uk)
 * @version 1.0
 * @since 2017-10-08
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

public class Learning3_2 {
	
	public static void main(String[] args) {
		//initializing the first game
		
		Learning3_2 l3 = new Learning3_2();
		ArrayList<String> nameList = new ArrayList<String>(){{add("Alex"); add("Alfred"); add("Anita"); add("Anne"); add("Bernard");}};
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
		ArrayList<ArrayList<Double>> inputList = new ArrayList<ArrayList<Double>>();
		inputList = initializeInput(inputList);
		ArrayList<ArrayList<Double>> outputList = new ArrayList<ArrayList<Double>>();
		outputList = initializeOutput(outputList);
		//play the game
		l3.play(nameList, questionList, inputList, outputList);
		
	}
	
	/**
	* this method is used to play a new game with given data.
	* @param 	nameList
	* @param	questionList
	* @param	inputList
	* @param	outputList
	*/
	public void play(ArrayList<String> nameList,ArrayList<String> questionList,  ArrayList<ArrayList<Double>> inputList, ArrayList<ArrayList<Double>> outputList) {
		//loading the network
		
		System.out.println("Loading network");
		BasicNetwork loaded_net = this.Learning(inputList, outputList, 3);
		//initializing the game
		double[] INPUT=new double[questionList.size()];
		ArrayList<Integer> wrongGuess = new ArrayList<Integer>();
		boolean flag = false;
		//whenever maybe is guessed, add that feature to the "maybe" arraylist
		ArrayList<Integer> maybe = new ArrayList<Integer>();
		int num = nameList.size();
		//start playing
		System.out.println("Let's play the guessWho game! \nPlease answer \"yes\", \"no\" or \"maybe\" for each question");
		
		for (int i = 0; i < questionList.size(); i++) {
			INPUT[i] = Judge(questionList.get(i));
			//if the answer is maybe, this question will be treated as the other unguessed ones in early guess part.
			if (INPUT[i] == -1) {
				maybe.add(i);
				INPUT[i] = 0.5;
			}
			/** Early guessing
			*	input1 assumes the answers for the rest questions and maybe questions are yes
			*	input2 assumes the answers for the rest questions and maybe questions are no
			*	if the guessing for the two inputs are identical and the guess wasn't on the wrong guess list
			*	make early guess
			*/
			
			double[] positiveINPUT = new double[questionList.size()];
			double[] negativeINPUT = new double[questionList.size()];
			for (int j = 0; j <= i; j++) {
				positiveINPUT[j] = INPUT[j];
				negativeINPUT[j] = INPUT[j];
			}
			
			for (int j = i + 1; j < questionList.size(); j++) {
				positiveINPUT[j] = 1.0;
				negativeINPUT[j] = 0.0;
			}
			
			for (int j = 0; j < questionList.size(); j++) {
				if (maybe.contains(j)) {
					positiveINPUT[j] = 1.0;
					negativeINPUT[j] = 0.0;
				}
			}
			
			MLData positiveInput=new BasicMLData(positiveINPUT);
			MLData positiveOutput = loaded_net.compute(positiveInput);
			MLData negativeInput=new BasicMLData(negativeINPUT);
			MLData negativeOutput = loaded_net.compute(negativeInput);
			
			//an extra condition >0.5 as threshold: see method
			if(Find(positiveOutput, num) == Find(negativeOutput, num) && Find(positiveOutput, num) != -1 && Find(negativeOutput, num) != -1 && !wrongGuess.contains(Find(positiveOutput, num))) {
				System.out.println("I am guessing " + nameList.get(Find(positiveOutput, num)));
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
			    	wrongGuess.add(Find(positiveOutput, num));
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
			int[] rank = Rank(output, num);
			int index = 0;
			String s = "no";
			//the agent then guess the most possible persons one by one sorted by the output value.
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
			if(s.equals("no")) {
				//ask if the player want to add his person to the database
				System.out.println("Sorry this character is not recorded. \nDo you want to add this one to the dataset with these features?");
				Scanner in = new Scanner(System.in);
			    s = in.nextLine();
			    if(s.equals("yes")) {
			    	//addChara
			    	addChara(nameList, questionList, inputList, outputList, INPUT, maybe);
			    }
			    else
			    {
			    	System.out.println("Okay, fancy another game?");
			    }
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
	    while( !s.equals("no") && !s.equals("yes") && !s.equals("maybe")) {
	    	System.out.println("Please answer \"yes\", \"no\" or \"maybe\" for each question.");
	    	in = new Scanner(System.in);
		    s = in.nextLine();
	    }
		if (s.equals("yes"))  ans = 1;
		else if(s.equals("maybe")) ans = -1;
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
			System.out.println(train.getError());
//			epoch++;
		   } while(train.getError() > 0.01);
		train.finishTraining();
		train.finishTraining();
		return network;
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
	
	/**
	* this method is used to update the input database.
	* @param 	maybe records which questions that the player answered maybe
	* @param	inputList is the original input database.
	* @param	INPUT records the players answers ("maybe"s are treated as "no"s here)
	* @return	updated input database.
	*/
	protected static ArrayList<ArrayList<Double>> addInput(ArrayList<Integer> maybe, ArrayList<ArrayList<Double>> inputList, double[] INPUT) {
		
		int more = (int) Math.pow(2, maybe.size());
		
		double[][] inputArray = new double[inputList.size() + more][inputList.get(0).size()];
		for(int i = 0; i < inputList.size(); i++) {
			for(int j = 0; j < inputList.get(0).size(); j++) {
				inputArray[i][j] = inputList.get(i).get(j);
			}
		}
		for(int i = 0; i < INPUT.length; i++) {
			inputArray[inputList.size()][i] = INPUT[i];
		}
		
		int[] maybeArray = new int[maybe.size()];
		for(int i = 0; i < maybe.size(); i++) {
			maybeArray[i] = maybe.get(i);
		}
		
		
		for(int i = inputList.size() + 1; i < inputList.size() + more; i++) {
			for(int j = 0; j < INPUT.length; j++) {
				inputArray[i][j] = inputArray[i-1][j];
			}
			int j = 0;
			inputArray[i][maybeArray[j]]++;
			while(inputArray[i][maybeArray[j]]==2) {
				inputArray[i][maybeArray[j]] -= 2;
				j = j + 1;
				inputArray[i][maybeArray[j]]++;
			}
		}
		inputList = writeList(inputArray);
		return inputList;
	}

	/**
	* this method is used to add a new character and then play a game.
	* @param 	nameList
	* @param	questionList
	* @param	inputList
	* @param	outputList
	*/
	protected static void addChara(ArrayList<String> nameList, ArrayList<String> questionList,  ArrayList<ArrayList<Double>> inputList, ArrayList<ArrayList<Double>> outputList, double[] INPUT, ArrayList<Integer> maybe) {
		//updating name database
		System.out.println("Please input the name of this person");
    	Scanner in = new Scanner(System.in);
    	String s = in.nextLine();
	    nameList.add(s);
	    
	    //updating output database
	    for(int j = 0; j < outputList.size(); j++) {
	    	outputList.get(j).add(0.0);
	    }
	    ArrayList<Double> outputTemp = new ArrayList<Double>();
    	for(int j = 0; j < nameList.size() - 1; j++) {
    		outputTemp.add(0.0);
    	}
    	outputTemp.add(1.0);
    	for(int j = 0; j < Math.pow(2, maybe.size()); j++) {
    		outputList.add(outputTemp);
    	}
    	//update input database
    	inputList = addInput(maybe, inputList, INPUT);
    	
    	//play a new game with updated database
    	Learning3_2 l3 = new Learning3_2();
        l3.play(nameList, questionList, inputList, outputList);
	}
	
}


