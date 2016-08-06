package io.inp;

import java.util.*;

public class QLearning {

    private boolean isRunning;

    private double gamma; // Discount Factor
    private double epsilon; // Probability of exploration and ( 1 - epsilon ) for exploitation.

    // Starting location is D,
    // Goal state is C with an immediate reward of +100.
    // Terminal state I is a -100 immediate reward tile.
    public enum State {
        A, B, C,
        D, E, F,
        G, H, I
    }

    public enum Action {
        UP, DOWN,
        LEFT, RIGHT
    }

    // Takes S x A -> S
    // Returns the next state given ( S, A ).
    private LinkedHashMap< State, LinkedHashMap< Action, State > > stateLinks;

    // qValues representing each State.
    // State A, UP, DOWN, LEFT, and RIGHT. These represent the set of A that is applicable for S.
    // Used in calculating maxQ( s, a ) for updating Q Value.
    private LinkedHashMap< State, LinkedHashMap< Action, Double > > qValues;

    // Immediate Reward Values for State S. Rt
    private LinkedHashMap< State, Double > rValues;

    private Random random;
    final private State startState = State.D;
    private State currentState;

    private int episodes;
    private int explore;

    private double final_epsilon;
    private double annealRate;

    public QLearning( double init_epsilon, double final_epsilon, int explore, int episodes, double gamma) {

        this.epsilon = Math.min( Math.max( 0, init_epsilon ), 1 ); // Constrain epsilon, 0 <= epsilon <= 1 ;
        this.final_epsilon = Math.min( Math.max( 0, final_epsilon ), 1 );
        this.episodes = Math.min( Math.max( 0, episodes ), 100000000 );

        this.explore = Math.min( Math.max( 5000, explore ), this.episodes );

        this.annealRate = ( this.epsilon - this.final_epsilon ) / this.episodes;

        this.gamma = Math.min( Math.max( 0, gamma ), 1 ); // Constrain alpha within 0 <= alpha <= 1



        this.random = new Random();

        this.qValues = new LinkedHashMap<>();
        this.rValues = new LinkedHashMap<>();

        for( State s : State.values() ){
            this.rValues.put( s, 0.0 );
            LinkedHashMap< Action, Double > expectedRewards = new LinkedHashMap<>();
            for( Action a : Action.values() ) {
                expectedRewards.put( a, 0.0 ); // For each direction, initialise expected Rewards to 0.
            }
            this.qValues.put( s, expectedRewards );
        }
        this.rValues.put( State.C, 100.0 );
        this.rValues.put( State.I, -100.0 );

        this.currentState = this.startState;

        setupGridWorld();

    }

    // Q(st,at) = rt+1 + γ maxat+1 Q(st+1,at+1)
    public void run(){
        if( !this.isRunning ){
            this.isRunning = true;

            for( int i = 0; i < this.episodes; ++i ) {

                this.currentState = this.startState;
                while ( this.currentState != State.C && this.currentState != State.I ) { // Account for terminal states.
                    // Epsilon-Greedy
                    // Action Selection
                    double rand = random.nextDouble();
                    Action a;
                    if ( i > this.explore && rand >= this.epsilon) {
                        // Exploitation
                        a = selectActionMax(this.currentState);
                    } else {
                        // Exploration
                        a = selectActionAtRandom(); // Ensure we have sufficient exploration before following sub-optimal policies.
                    }

                    State nextS = stateLinks.get( this.currentState ).get( a );
                    //double currentQ = qValues.get( this.currentState ).get( a );

                    double maxQValue = Collections.max( qValues.get( nextS ).values() ); // Get the max Q value from next State.
                    double immediateReward = rValues.get( nextS );

                    double updatedQ = immediateReward + ( this.gamma * maxQValue );
                    setQ( this.currentState, a, updatedQ );

                    this.currentState = nextS;

                    // Only anneal if we're out of the initial exploration stages.
                    if( i > this.explore ) {
                        annealEpsilon();
                    }
                }

                if( i % 10000 == 0 ){ // Every X episodes.
                    System.out.println( "Current Episode: " + i );
                    printMaxQValues();
                    printPolicy();
                }
            }

            this.isRunning = false;
        }
    }

    private void setQ( State s, Action a, Double q ){
        qValues.get( s ).put( a, q );
    }

    private void annealEpsilon(){
        if( this.epsilon > this.final_epsilon ) {
            this.epsilon -= this.annealRate;
        }
    }

    public Action selectActionAtRandom(){
        return Action.class.getEnumConstants()[ random.nextInt( Action.class.getEnumConstants().length ) ];
    }

    public Action selectActionMax( State s ){
        LinkedHashMap< Action, Double > actions = qValues.get( s );
        Double max = Collections.max( actions.values() );

        List< Action > actionArray = new LinkedList<>();

        actions.forEach( ( k, v ) -> {
            if( v.equals( max ) ){
                actionArray.add( k );
            }
        } );

        return actionArray.get( random.nextInt( actionArray.size() ) );

    }

    public void printMaxQValues(){
        int i = 1;
        for( State s : State.class.getEnumConstants() ){
            System.out.print( Collections.max( qValues.get( s ).values() ) + ",\t\t" );
            if( i % 3 == 0 ){
                System.out.println(); // New Line!
            }
            ++i;
        }
        System.out.println(); // Spacer
    }

    public void printPolicy(){
        int i = 1;
        for( State s : State.class.getEnumConstants() ){
            System.out.print( selectActionMax( s ) + ",\t\t" );
            if( i % 3 == 0 ){
                System.out.println(); // New Line!
            }
            ++i;
        }
        System.out.println(); // Spacer
    }

    private void setupGridWorld(){

        stateLinks = new LinkedHashMap<>();

        // State A
        LinkedHashMap< Action, State > stateTransfers;

        stateTransfers = new LinkedHashMap< Action, State >();
        stateTransfers.put( Action.UP, State.A );
        stateTransfers.put( Action.DOWN, State.D );
        stateTransfers.put( Action.LEFT, State.A );
        stateTransfers.put( Action.RIGHT, State.B );

        stateLinks.put( State.A, stateTransfers );

        // State B
        stateTransfers = new LinkedHashMap< Action, State >();
        stateTransfers.put( Action.UP, State.B );
        stateTransfers.put( Action.DOWN, State.E );
        stateTransfers.put( Action.LEFT, State.A );
        stateTransfers.put( Action.RIGHT, State.C );

        stateLinks.put( State.B, stateTransfers );

        // State C
        stateTransfers = new LinkedHashMap< Action, State >();
        stateTransfers.put( Action.UP, State.C );
        stateTransfers.put( Action.DOWN, State.C );
        stateTransfers.put( Action.LEFT, State.C );
        stateTransfers.put( Action.RIGHT, State.C );

        stateLinks.put( State.C, stateTransfers );

        // State D
        stateTransfers = new LinkedHashMap< Action, State >();
        stateTransfers.put( Action.UP, State.A );
        stateTransfers.put( Action.DOWN, State.G );
        stateTransfers.put( Action.LEFT, State.D );
        stateTransfers.put( Action.RIGHT, State.E );

        stateLinks.put( State.D, stateTransfers );

        // State E
        stateTransfers = new LinkedHashMap< Action, State >();
        stateTransfers.put( Action.UP, State.B );
        stateTransfers.put( Action.DOWN, State.H );
        stateTransfers.put( Action.LEFT, State.D );
        stateTransfers.put( Action.RIGHT, State.F );

        stateLinks.put( State.E, stateTransfers );

        // State F
        stateTransfers = new LinkedHashMap< Action, State >();
        stateTransfers.put( Action.UP, State.C );
        stateTransfers.put( Action.DOWN, State.I );
        stateTransfers.put( Action.LEFT, State.E );
        stateTransfers.put( Action.RIGHT, State.F );

        stateLinks.put( State.F, stateTransfers );

        // State G
        stateTransfers = new LinkedHashMap< Action, State >();
        stateTransfers.put( Action.UP, State.D );
        stateTransfers.put( Action.DOWN, State.G );
        stateTransfers.put( Action.LEFT, State.D );
        stateTransfers.put( Action.RIGHT, State.H );

        stateLinks.put( State.G, stateTransfers );

        // State H
        stateTransfers = new LinkedHashMap< Action, State >();
        stateTransfers.put( Action.UP, State.E );
        stateTransfers.put( Action.DOWN, State.H );
        stateTransfers.put( Action.LEFT, State.G );
        stateTransfers.put( Action.RIGHT, State.I );

        stateLinks.put( State.H, stateTransfers );

        // State I
        stateTransfers = new LinkedHashMap< Action, State >();
        stateTransfers.put( Action.UP, State.I );
        stateTransfers.put( Action.DOWN, State.I );
        stateTransfers.put( Action.LEFT, State.I );
        stateTransfers.put( Action.RIGHT, State.I );

        stateLinks.put( State.I, stateTransfers );

    }

}
