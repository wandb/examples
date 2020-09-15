# Java `sin` function

This example uses the [Java Client](https://github.com/wandb/client-java) to plot a
sin function.

In order to run this example you must have Java JDK and Maven install on your
machine. You must also install the `wandb[grpc]` package and be authenticated by
calling `wandb login MY_API_KEY`.

For more information checkout the [Java Client
documentation](https://docs.wandb.com/java).

## Running the Example

1. Build the jar file by running `mvn package`.
2. Execute the jar file by running `java -jar ./target/wandb-sin-func-1.0-SNAPSHOT-jar-with-dependencies.jar`.
