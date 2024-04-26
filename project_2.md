# project 2: Order execution problem -- algo trading
In this project, our objective is to develop an order execution strategy and to conduct trading simulation using the provided data. The project is divided into three parts, which are to be completed step by step, making it easier for the candidates to achieve the goal.

The provided file TradeTaskLog.csv contains the selected execution tasks, with each line representing a 5-minute task. It also provides information of each task,  including the date and time the task begins, the contract to be traded, the direction of the trading task (buy or sell), the volume be traded, and the trading volume in the market during the task period. The level-2 data corresponding to each execution task would be provided. Researchers can use either Python or C/C++ to complete the data analysis and subsequent simulation.

## Part 1
In this part, researchers need to use level-2 data to infer the distribution of trading volumes between two snapshots, such as how many prices had trades executed and what the volume of trades was at each price. Researchers only need to infer trading events for the snapshots inside the 5-minute execution task. This part is the preparatory work for the subsequent simulation of execution strategies. 

Inference of trading events between snapshots is usually not entirely correct. We only require researchers to make reasonable estimates of the trade volume distribution in most cases where the inference is relatively straightforward. For example, when ask_price1 and bid_price1 do not change between two consecutive snapshots, we can assume that all trades are traded on either bid_price1 or ask_price1. Then we can use the trade volume and trade amount between two consecutive snapshots to deduce the trade volumes at the bid price 1 and ask price 1 (by solving linear equations, in this case a set of two equations with two variables).

For more complex situations where bid1 and ask1 have changed between snapshots, we can calculate the VWAP between the two snapshots and then make the assumption that all trades occurred at the two price levels closest to the VWAP. By making assumptions about trade distribution, we can again solve linear equations to estimate trade volumes at specific price levels.

## Part 2
Simulate a TWAP execution strategy, where we only place passive orders at the best bid or best ask price. For each execution task, researchers need to equally divide the total volume into every 30-second intervals during the 5-minute task. Assuming that the orders you place are at the end of the queue for the current ask1 or bid1, you need to monitor and update the queue by using the trading events calculated in part 1. If the current order is no longer at the best bid or best ask, we need to cancel the order and re-place it at the best bid or best ask.

By performing the simulation, calculate the percentage of the strategy's traded volume relative to the target volume of the task for each execution task. Then try to modify the TWAP execution strategy, so that the ratio could be higher than 95%.

## Part 3





