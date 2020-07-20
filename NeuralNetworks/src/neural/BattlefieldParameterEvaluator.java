package neural;

import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.Random;
import javax.imageio.ImageIO;

import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.neural.data.basic.BasicNeuralDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;

import robocode.control.*;
import robocode.control.events.*;

public class BattlefieldParameterEvaluator {
	
	// Minimum allowable battlefield size is 400
	final static int MAXBATTLEFIELDSIZE = 800;
	
	// Minimum allowable gun cooling rate is 0.1
	final static double MAXGUNCOOLINGRATE = 0.5;
	
	final static int NUMBATTLEFIELDSIZES = 601;
	final static int NUMCOOLINGRATES = 501;
	final static int NUMSAMPLES = 1000;
	
	// Number of inputs for the multilayer perceptron (size of the input vectors)
	final static int NUM_NN_INPUTS = 2;
	
	// Number of hidden neurons of the neural network
	final static int NUM_NN_HIDDEN_UNITS = 50;
	
	// Number of epochs for training
	final static int NUM_TRAINING_EPOCHS = 100000;
	
	static int NdxBattle;
	static double[] NumTurns;
	
	
	public static void main(String[] args) {
		
		double[] BattlefieldSize = new double[NUMSAMPLES];
		double[] GunCoolingRate = new double[NUMSAMPLES];
		NumTurns = new double[NUMSAMPLES];
		Random rng = new Random(15L);
		
		// Disable log messages from Robocode
		RobocodeEngine.setLogMessagesEnabled(false);
		
		
		/** Create the RobocodeEngine **/
		// Run from C:/Robocode
		RobocodeEngine engine = new RobocodeEngine(new java.io.File("C:/Robocode"));
		
		// Add our own battle listener to the RobocodeEngine
		engine.addBattleListener(new BattleObserver());
		
		// Show the Robocode battle view
		engine.setVisible(true);
		
		/** Setup the battle specification **/
		// Setup battle parameters
		int numberOfRounds = 1;
		long inactivityTime = 100;
		int sentryBorderSize = 50;
		boolean hideEnemyNames = false;
		int NumObstacles;
		
		// Get the robots and set up their initial states
		RobotSpecification[] modelRobots = engine.getLocalRepository("sample.TrackFire, sample.SittingDuck");
		RobotSpecification[] competingRobots;
		RobotSetup[] robotSetups;
		
		
		for (NdxBattle = 0; NdxBattle < NUMSAMPLES; NdxBattle++) {
			
			// Choose the battlefield size and gun cooling rate
			BattlefieldSize[NdxBattle] = MAXBATTLEFIELDSIZE*(0.5+0.5*rng.nextDouble());
			GunCoolingRate[NdxBattle] = MAXGUNCOOLINGRATE*(0.2+0.8*rng.nextDouble());
			
			// Create the battlefield
			BattlefieldSpecification battlefield = new BattlefieldSpecification((int)BattlefieldSize[NdxBattle],
																					(int)BattlefieldSize[NdxBattle]);
			
			// Set the robot positions
			NumObstacles = (int)(0.00001 * (int)BattlefieldSize[NdxBattle] * (int)BattlefieldSize[NdxBattle]);
			robotSetups = new RobotSetup[NumObstacles+1];
			competingRobots = new RobotSpecification[NumObstacles+1];
			
			robotSetups[0] = new RobotSetup(2*BattlefieldSize[NdxBattle]/3.0, BattlefieldSize[NdxBattle]/2.0, 0.0);
			competingRobots[0] = modelRobots[0];
			
			for (int i = 1; i <= NumObstacles; i++) {
				robotSetups[i] = new RobotSetup(BattlefieldSize[NdxBattle]/3.0, i*BattlefieldSize[NdxBattle]/(NumObstacles+1), 0.0);
				competingRobots[i] = modelRobots[1];
			}
			
			// Prepare the battle specification
			BattleSpecification battleSpec = new BattleSpecification(battlefield, numberOfRounds, inactivityTime,
																		GunCoolingRate[NdxBattle], sentryBorderSize,
																			hideEnemyNames, competingRobots, robotSetups);
			
			// Run our specified battle and let it run till it is over
			engine.runBattle(battleSpec, true); // waits till the battle finishes
		}
		
		/** Cleanup our RobocodeEngine **/
		engine.close();
		
		/** Print results for battle **/
		System.out.println("*******************************");
		System.out.println("All battles completed");
		System.out.println("*******************************");
		System.out.println("Data collected:");
		System.out.println("-------------------------------");
		for (int i = 0; i < NUMSAMPLES; i++) {
			System.out.println("Battle " + (i+1) + " specifications");
			System.out.println("  (x1) Battlefield size: " + BattlefieldSize[i] + " x " + BattlefieldSize[i]);
			System.out.println("  (x2) Gun cooling rate: " + GunCoolingRate[i]);
			System.out.println("   (y) Number of turns needed: " + NumTurns[i]);
		}
		
		/** Create the training dataset for the neural network **/
		double[][] RawInputs = new double[NUMSAMPLES][NUM_NN_INPUTS];
		double[][] RawOutputs = new double[NUMSAMPLES][1];
		
		for (int NdxSample = 0; NdxSample < NUMSAMPLES; NdxSample++) {
			// IMPORTANT: normalize the inputs and the outputs to the interval [0,1]
			RawInputs[NdxSample][0] = BattlefieldSize[NdxSample]/MAXBATTLEFIELDSIZE;
			RawInputs[NdxSample][1] = GunCoolingRate[NdxSample]/MAXGUNCOOLINGRATE;
			RawOutputs[NdxSample][0] = NumTurns[NdxSample]/1000;
		}
		
		BasicNeuralDataSet MyDataSet = new BasicNeuralDataSet(RawInputs, RawOutputs);
		
		/** Create and train the neural network **/
		// Create Feed Forward Network
		BasicNetwork ffNetwork = new BasicNetwork();
		ffNetwork.addLayer(new BasicLayer(null,true, NUM_NN_INPUTS));
		ffNetwork.addLayer(new BasicLayer(new ActivationSigmoid(), true, NUM_NN_HIDDEN_UNITS));
		ffNetwork.addLayer(new BasicLayer(new ActivationSigmoid(), false, 1));
		ffNetwork.getStructure().finalizeStructure();
		ffNetwork.reset();
		
		// Train the NeuralNetwork
		System.out.println("Training network...");
		
		final ResilientPropagation train = new ResilientPropagation(ffNetwork, MyDataSet);
	    for (int epoch = 0; epoch < NUM_TRAINING_EPOCHS; epoch++) {
	        train.iteration();
	        System.out.println("Epoch #" + (epoch+1) + "   Error: " + train.getError());
	    }
	    train.finishTraining();
		
		System.out.println("Training completed.");
		
		System.out.println("Testing network...");
		
		/** Generate test samples to build an output image **/
		int[] OutputRGBint = new int[NUMBATTLEFIELDSIZES*NUMCOOLINGRATES];
		Color MyColor;
		double MyValue = 0;
		double[][] MyTestData = new double[NUMBATTLEFIELDSIZES*NUMCOOLINGRATES][NUM_NN_INPUTS];
		
		for (int NdxBattleSize = 0; NdxBattleSize < NUMBATTLEFIELDSIZES; NdxBattleSize++) {
			for (int NdxCooling = 0; NdxCooling < NUMCOOLINGRATES; NdxCooling++) {
				MyTestData[NdxCooling+NdxBattleSize*NUMCOOLINGRATES][0] = 0.1+0.9*((double)NdxBattleSize)/NUMBATTLEFIELDSIZES;
				MyTestData[NdxCooling+NdxBattleSize*NUMCOOLINGRATES][1] = 0.1+0.9*((double)NdxCooling)/NUMCOOLINGRATES;
			}
		}
		
		// Simulate the neural network with the test samples and fill a matrix
		for (int NdxBattleSize = 0; NdxBattleSize < NUMBATTLEFIELDSIZES; NdxBattleSize++) {
			for(int NdxCooling = 0; NdxCooling < NUMCOOLINGRATES; NdxCooling++) {
				double input[] = MyTestData[NdxCooling + NdxBattleSize*NUMCOOLINGRATES];
				final BasicMLData input2 = new BasicMLData(input);
				double MyResult = ffNetwork.compute(input2).getData(0);
				MyValue = ClipColor(MyResult);
				MyColor = new Color((float)MyValue, (float)MyValue, (float)MyValue);
				OutputRGBint[NdxCooling+NdxBattleSize*NUMCOOLINGRATES] = MyColor.getRGB();
			}
		}
		System.out.println("Testing completed.");
		
		/** Plot the training samples **/
		for (int NdxSample = 0; NdxSample < NUMSAMPLES; NdxSample++) {
			MyValue = ClipColor(NumTurns[NdxSample]/1000);
			MyColor = new Color((float)MyValue, (float)MyValue, (float)MyValue);
			int MyPixelIndex = (int)(Math.round(NUMCOOLINGRATES * ((GunCoolingRate[NdxSample]/MAXGUNCOOLINGRATE)-0.1) / 0.9)
					+ Math.round(NUMBATTLEFIELDSIZES*((BattlefieldSize[NdxSample]/MAXBATTLEFIELDSIZE)-0.1)/0.9)*NUMCOOLINGRATES);
			
			if ((MyPixelIndex>=0) && (MyPixelIndex<NUMCOOLINGRATES*NUMBATTLEFIELDSIZES)) {
				OutputRGBint[MyPixelIndex] = MyColor.getRGB();
			}
		}
		
		BufferedImage img = new BufferedImage(NUMCOOLINGRATES, NUMBATTLEFIELDSIZES, BufferedImage.TYPE_INT_RGB);
		img.setRGB(0, 0, NUMCOOLINGRATES, NUMBATTLEFIELDSIZES, OutputRGBint, 0, NUMCOOLINGRATES);
		File f = new File("hello.png");
		try {
			ImageIO.write(img, "png", f);
		} catch (IOException e) {
			e.printStackTrace();
		}
		System.out.println("Image generated.");
		
		// Make sure that the Java VM is shut down properly
		System.exit(0);
	}
	
	/**
	* Clip a color value (double precision) to lie in the valid range [0,1]
	**/
	public static double ClipColor(double Value) {
		if (Value < 0.0) {
			Value = 0.0;
		}
		if (Value > 1.0) {
			Value = 1.0;
		}
		return Value;
	}

	/**
	* Our private battle listener for handling the battle event we are interested in.
	**/
	static class BattleObserver extends BattleAdaptor {
		
		// Called when the game sends out an information message during the battle
		public void onBattleMessage(BattleMessageEvent e) {
			//System.out.println("Msg> " + e.getMessage());
		}
		
		// Called when the game sends out an error message during the battle
		public void onBattleError(BattleErrorEvent e) {
			System.out.println("Err> " + e.getError());
		}
		
		// Called when a round ends
		public void onRoundEnded(RoundEndedEvent e) {
			// Store the number of turns of the robot			
			System.out.println("*******************************");
			System.out.println("---   Battle " + (NdxBattle+1) + " completed   ---");
			System.out.println("-------------------------------");
			System.out.println("Number of turns needed: " + e.getTurns());
			System.out.println("*******************************");
			
			BattlefieldParameterEvaluator.NumTurns[NdxBattle] = e.getTurns();
		}
	}
}