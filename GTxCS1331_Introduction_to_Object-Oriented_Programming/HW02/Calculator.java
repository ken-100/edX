import java.util.Scanner;
public class Calculator{
	
	public static void main(String[] args){
	    Scanner scanner = new Scanner(System.in);

	    System.out.print("List of operations: add subtract multiply divide alphabetize\r\n");
	    System.out.print("Enter an operation:");
	    String method = scanner.nextLine();
	   
	    double a=0;
	    double b=0;
	    int a_i=0;
	    int b_i=0;
	    String c = "";
	    String d = "";
	    
	    if (method.toLowerCase().equals("alphabetize")) {
	    	
		    System.out.print("Enter two words:");    
		    
		    c = scanner.nextLine();	    
		    d = scanner.nextLine();
		    
	    	System.out.println("Enter two words:");
	    	System.out.println(c + " " + d);
		    
		    if (c.toUpperCase().equals(d.toUpperCase())) {
		    	System.out.println("Chicken or Egg." );
		    	
		    }else {
		    	
		    	int tmp = c.compareTo(d);
		        if (tmp > 0) {
		            System.out.println("Answer: " + d + " comes before " + c + " alphabetically.");
		        } else {
		            System.out.println("Answer: " + c + " comes before " + d + " alphabetically.");
		        } 
	        
	    	}
		  
	    }else {
	    
		    double o = 0;
		    int o_i = 0;
		    
		    switch (method.toLowerCase()) {
		    case "add":
		    	
		    	try {
				    System.out.println("Enter two integers:");
				    
					a = scanner.nextDouble();	    
					b = scanner.nextDouble();
					a_i = (int) a;
					b_i = (int) b;   
					
//					System.out.println("Enter an operation:");
//					System.out.println(method);
					System.out.println("Enter two integers:");
					
					if ( a_i - a == 0 && b_i - b == 0 ) {
						o_i = a_i + b_i ;     
						System.out.println(a_i + " " + b_i);	
						System.out.println("Answer: " + o_i);
				    }else {
				    		System.out.println(a + " " + b);
				    		System.out.println("Invalid input entered. Terminating...");
				    }
		    	}
		    	catch(Exception e) {
		    		System.out.println("Enter two integers:");
		    		System.out.println("Hi String");
		    		System.out.println("Invalid input entered. Terminating...");
		    	}

		      break;		
		      
		    case "subtract":
		    	try {
					a = scanner.nextDouble();	    
					b = scanner.nextDouble();
					a_i = (int) a;
					b_i = (int) b;
					System.out.println("Enter two integers:");
					
					if ( a_i - a == 0 && b_i - b == 0 ) {
						o_i = a_i - b_i ;     
						System.out.println(a_i + " " + b_i);	
						System.out.println("Answer: " + o_i);
					}else {
						System.out.println(a + " " + b);
						System.out.println("Invalid input entered. Terminating...");
						    }
		    	}
		    	catch(Exception e) {
		    		System.out.println("Enter two integers:");
		    		System.out.println("Hi String");
		    		System.out.println("Invalid input entered. Terminating...");
		    	}
				
		      break;
		    case "multiply":
				
		    	try {
		    		a = scanner.nextDouble();	    
		    		b = scanner.nextDouble();
					a_i = (int) a;
					b_i = (int) b;
					
			    	o = a * b;
				    System.out.println("Enter two doubles:");
				    
					if ( a_i - a == 0 && b_i - b == 0 ) {
						System.out.println(a_i + " " + b_i); 
				    }else {
				    System.out.println(a + " " + b); 	
				    }
			    	System.out.printf("Answer: " + String.format( "%.2f",(double)Math.round(o * 100)/100));
		    	}
		    	catch(Exception e) {
		    		System.out.println("Enter two doubles:");
		    		System.out.println("Hi String");
		    		System.out.println("Invalid input entered. Terminating...");
		    		a = 1;
		    		b = 1;
		    	}
		      break;
		      
		    case "divide":
				
		    	try {
		    		a = scanner.nextDouble();	    
		    		b = scanner.nextDouble();
		    		
					a_i = (int) a;
					b_i = (int) b;
			    	o = a / b;
				    System.out.println("Enter two doubles:");

					if ( a_i - a == 0 && b_i - b == 0 ) {
						System.out.println(a_i + " " + b_i); 
				    }else {
				    System.out.println(a + " " + b); 	
				    }
				    
				    if ( b == 0  ) {
				    	System.out.println("Invalid input entered. Terminating...");
				    }else {
				    	System.out.printf("Answer: " + String.format( "%.2f",(double)Math.round(o * 100)/100));
			    	}				    
		    	}
		    	catch(Exception e) {
		    		System.out.println("Enter two doubles:");
		    		System.out.println("Hi String");
		    		System.out.println("Invalid input entered. Terminating...");
		    	}

		      break;
		      
		    default:
			    System.out.println("Invalid input entered. Terminating...");
		    		    
	    }
    	
	    scanner.close();
	
	  }
	}
}
