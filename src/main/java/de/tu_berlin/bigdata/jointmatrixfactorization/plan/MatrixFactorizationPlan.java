/*
 * Project: JointMatrixFactorization
 * @author Fangzhou Yang
 * @author Xugang Zhou
 * @version 1.0
 */

package de.tu_berlin.bigdata.jointmatrixfactorization.plan;

import de.tu_berlin.dima.bigdata.jointmatrixfactorization.mapper.InitItemFeatureVectorReducer;
import de.tu_berlin.dima.bigdata.jointmatrixfactorization.mapper.TuppleMapper;
import de.tu_berlin.dima.bigdata.jointmatrixfactorization.predict.PredictionCrosser;
import de.tu_berlin.dima.bigdata.jointmatrixfactorization.solve.ItemFeatureVectorUpdateReducer;
import de.tu_berlin.dima.bigdata.jointmatrixfactorization.solve.Joint;
import de.tu_berlin.dima.bigdata.jointmatrixfactorization.solve.UserFeatureVectorUpdateReducer;
import de.tu_berlin.dima.bigdata.jointmatrixfactorization.util.Util;
import eu.stratosphere.pact.common.contract.CrossContract;
import eu.stratosphere.pact.common.contract.FileDataSink;
import eu.stratosphere.pact.common.contract.FileDataSource;
import eu.stratosphere.pact.common.contract.MapContract;
import eu.stratosphere.pact.common.contract.MatchContract;
import eu.stratosphere.pact.common.contract.ReduceContract;
import eu.stratosphere.pact.common.io.RecordOutputFormat;
import eu.stratosphere.pact.common.io.TextInputFormat;
import eu.stratosphere.pact.common.plan.Plan;
import eu.stratosphere.pact.common.plan.PlanAssembler;
import eu.stratosphere.pact.common.plan.PlanAssemblerDescription;
import eu.stratosphere.pact.common.type.base.PactFloat;
import eu.stratosphere.pact.common.type.base.PactInteger;

/*
 * This Class is the the "plan" class of this project.
 */
public class MatrixFactorizationPlan implements PlanAssembler, PlanAssemblerDescription{
	
	private final int numIterations = 50;
	
	private final MatchContract userFeatureVectorUpdateJoints[] = new MatchContract[numIterations];
	private final MatchContract itemFeatureVectorUpdateJoints[] = new MatchContract[numIterations];
	private final ReduceContract userFeatureVectorUpdateReducers[] = new ReduceContract[numIterations];
	private final ReduceContract itemFeatureVectorUpdateReducers[] = new ReduceContract[numIterations];
	
	  /*
	   * This method defines how the data would be operated.
	   * @return The whole scala-plan
	   * @param args(0) Path to input file
	   * @param args(1) Path to output file
	   * @param args(2) Number of subtasks to specify parallelism
	   */
	@Override
	public String getDescription() {
		return "Usage: [inputPath] [outputPath] ([numSubtasks])";
	}
	@Override
	public Plan getPlan(String... args) {
		String inputPath = args.length >= 1 ? args[0] : "";
		String outputPath = args.length >= 2 ? args[1] : "";
		int numSubtasks = args.length >= 3 ? Integer.parseInt(args[2]) : 1;

		FileDataSource source = new FileDataSource(new TextInputFormat(), inputPath, "Input Documents");

		System.out.println("Processing.. start iteration..");
		
		MapContract tuppleMapper = MapContract
				.builder(TuppleMapper.class).input(source)
				.name("Rating Tupple Mapper").build();

	    /*
	     * Initialize item-feature-vectors with random value
	     */
		ReduceContract initItemFeatureVectorReducer = ReduceContract
				.builder(InitItemFeatureVectorReducer.class, PactInteger.class, 1)
				.input(tuppleMapper).name("Init Item Feature Vector Reducer").build();
		
		
	    /*
	     * Learn the user-feature-vectors with initialized item-feature-vectors
	     */
		userFeatureVectorUpdateJoints[0] = MatchContract
				.builder(Joint.class, PactInteger.class, 1, 0)
				.input1(tuppleMapper)
				.input2(initItemFeatureVectorReducer)
				.name("user Feature Vector Update Joint 0")
				.build();
		
		userFeatureVectorUpdateReducers[0] = ReduceContract
				.builder(UserFeatureVectorUpdateReducer.class, PactInteger.class, 0)
				.input(userFeatureVectorUpdateJoints[0])
				.name("user Feature Vector Update Reducer 0")
				.build();
		
		
	    /*
	     * Continue to Alternative-Least-Sqaure (ALS) learning with numIter iterations
	     */
		for(int i = 1; i < numIterations; i ++){
		    /*
		     * Learn the item-feature-vectors with user-feature-vectors
		     */
			itemFeatureVectorUpdateJoints[i-1] = MatchContract
					.builder(Joint.class, PactInteger.class, 0, 0)
					.input1(tuppleMapper)
					.input2(userFeatureVectorUpdateReducers[0])
					.name("item Feature Vector Update Joint " + (i-1))
					.build();
			itemFeatureVectorUpdateReducers[i-1] = ReduceContract
					.builder(ItemFeatureVectorUpdateReducer.class, PactInteger.class, 1)
					.input(itemFeatureVectorUpdateJoints[i-1])
					.name("item Feature Vector Update Reducer " + (i-1))
					.build();
			
			userFeatureVectorUpdateJoints[i] = MatchContract
					.builder(Joint.class, PactInteger.class, 1, 0)
					.input1(tuppleMapper)
					.input2(itemFeatureVectorUpdateReducers[i-1])
					.name("user Feature Vector Update Joint " + i)
					.build();
			
			userFeatureVectorUpdateReducers[i] = ReduceContract
					.builder(UserFeatureVectorUpdateReducer.class, PactInteger.class, 0)
					.input(userFeatureVectorUpdateJoints[i])
					.name("user Feature Vector Update Reducer " + i)
					.build();
			
		}
		
		itemFeatureVectorUpdateJoints[numIterations-1] = MatchContract
				.builder(Joint.class, PactInteger.class, 0, 0)
				.input1(tuppleMapper)
				.input2(userFeatureVectorUpdateReducers[numIterations-1])
				.name("item Feature Vector Update Joint " + (numIterations-1))
				.build();
		itemFeatureVectorUpdateReducers[numIterations-1] = ReduceContract
				.builder(ItemFeatureVectorUpdateReducer.class, PactInteger.class, 1)
				.input(itemFeatureVectorUpdateJoints[numIterations-1])
				.name("item Feature Vector Update Reducer " + (numIterations-1))
				.build();
		
	    /*
	     * Use learned user- and item-feature-vectors to do the prediction of rating
	     */
		CrossContract predictCrosser = CrossContract.builder(PredictionCrosser.class)
				.input1(itemFeatureVectorUpdateReducers[numIterations-1])
				.input2(userFeatureVectorUpdateReducers[numIterations-1])
				.name("Predict Crosser")
				.build();
		
	    /*
	     * Put the predicted-rating result to output stream
	     */
		FileDataSink sink = new FileDataSink(RecordOutputFormat.class, outputPath, predictCrosser, "Rating Prediction");
		RecordOutputFormat.configureRecordFormat(sink)
			.recordDelimiter('\n')
			.fieldDelimiter(' ')
			.field(PactInteger.class, 0)
			.field(PactInteger.class, 1)
			.field(PactFloat.class, 2);
		

	    /*
	     * Return the plan
	     */
		Plan plan = new Plan(sink, "Rating Prediction Computation");
		plan.setDefaultParallelism(numSubtasks);

		return plan;
	}
	
	
	/*
	 * This object enables you to run this project locally.
	 * Run this object with the parameters specified below will result in run this project locally.
	 */
	public static void main(String[] args) throws Exception {

		String inputPath = "file://"+System.getProperty("user.dir") +"/datasets/100k/ua.base";

		String outputPath = "file://"+System.getProperty("user.dir") +"/results/100k/Prediction_ua_i=50.result";
		

		System.out.println("Reading input from " + inputPath);
		System.out.println("Writing output to " + outputPath);

		Plan toExecute = new MatrixFactorizationPlan().getPlan(inputPath, outputPath);
		Util.executePlan(toExecute);
	}
}