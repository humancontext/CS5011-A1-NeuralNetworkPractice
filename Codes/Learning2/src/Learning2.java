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

import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.neural.networks.BasicNetwork;
import org.encog.persist.EncogDirectoryPersistence;

public class Learning2 {
	
	public static void main(String[] args) {
		//initializing the quetion and name maps
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
		//loading the network
		System.out.println("Loading network");
		BasicNetwork loaded_net = (BasicNetwork) EncogDirectoryPersistence.loadObject(new File ("my_network.eg") ) ;
		Learning2 l2 = new Learning2();
		l2.play(nameList, questionList, loaded_net);
	}
		
	
	public void play(ArrayList<String> nameList, ArrayList<String> questionList, BasicNetwork loaded_net) {
		//initializing the game
		double[] INPUT=new double[questionList.size()];
		ArrayList<Integer> wrongGuess = new ArrayList<Integer>();
		boolean flag = false;
		int num = nameList.size();
		
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
			Scanner in = new Scanner(System.in);
		    String s = in.nextLine();
		    if(s.equals("yes")) {
		    	Learning2 l2 = new Learning2();
				l2.play(nameList, questionList, loaded_net);
		    }
		}
		else {
			//in case all early guessing does not work, guess the names one by one with greatest value for one-hot coding
			MLData data=new BasicMLData(INPUT);
			MLData output = loaded_net.compute(data);
			//rank by the output value for one-hot coding
			int[] rank = Rank(output, nameList.size());
			int index = 0;
//			for (int i = 0; i < 5; i++) {
//				System.out.print(output.getData(i) + ", ");
//			}
//			System.out.println();
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
			if(index > nameList.size()) System.out.println("Sorry this character is not recorded.");
			else {
				System.out.println("Great! Wanna try again?");
				Scanner in = new Scanner(System.in);
			    s = in.nextLine();
			    if(s.equals("yes")) {
			    	Learning2 l2 = new Learning2();
					l2.play(nameList, questionList, loaded_net);
			    }
			}
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
	    	System.out.println("Please answer yes or no.");
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
}


