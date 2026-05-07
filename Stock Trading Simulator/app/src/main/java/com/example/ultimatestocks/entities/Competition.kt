package com.example.ultimatestocks.entities

import kotlinx.serialization.Serializable

// A data class for a competition
@Serializable
data class Competition (
    val id: String = java.util.UUID.randomUUID().toString(),
    val title: String,
    val availableStocks: List<StockDetails>,
    var hasStarted: Boolean,
    var hasEnded: Boolean,
    val initialCash: Float,
    val players: MutableMap<String, Player>,
    val maxNewsEvents: Int,
    var newsEvents: MutableList<NewsEvent>,
    var latestNewsNotRevealed: String,
) {
    fun getStockValue(target: String): Float {
        for (stock in availableStocks) {
            if (stock.ticker == target) return stock.historicPrice.last()
        }
        return 0.0f
    }

    fun getLeaderboard() : List<Player> {
        val playerList = players.values.toList()
        val playerComparator = Comparator {p1: Player, p2: Player -> (p1.getPortfolioValue(this) - p2.getPortfolioValue(this)).toInt()}
        println(playerList)
        return playerList.sortedWith(playerComparator)
    }

    fun getPositionInLeaderboard(leaderboard: List<Player>, userId: String): Int {
        var playerPos = 100
        println(leaderboard)
        println(userId)
        leaderboard.forEachIndexed { index, player ->
            if (player.userID == userId) {
                playerPos = index
            }
        }
        println(playerPos)
        return playerPos
    }

    fun buy(ticker: String, numberOfShares: Int, playerId: String) {
        val playerInfo = players.getValue(playerId)
        val price = getStockValue(ticker)
        playerInfo.cash -= (price * numberOfShares)
        var added = false
        for (stock in playerInfo.portfolio) {
            if (stock.ticker == ticker) {
                stock.quantityOwned += numberOfShares
                added = true
            }
        }
        if (!added) playerInfo.portfolio.add(StockInventory(ticker, numberOfShares))
    }

    fun sell(ticker: String, numberOfShares: Int, playerId: String) {
        val playerInfo = players.getValue(playerId)
        val price = getStockValue(ticker)
        playerInfo.cash += (price * numberOfShares)
        var added = false
        for (stock in playerInfo.portfolio) {
            if (stock.ticker == ticker) {
                if (stock.quantityOwned > numberOfShares) {
                    stock.quantityOwned -= numberOfShares
                } else {
                    playerInfo.portfolio.remove(stock)
                }
                break
            }
        }
    }
}

@Serializable
data class Player(
    val userID: String,
    var cash: Float,
    var portfolio: MutableList<StockInventory>
) {
    fun getPortfolioValue(comp: Competition): Float {
        var stockVal = 0.0f
        for (stock in portfolio) {
            stockVal += (comp.getStockValue(stock.ticker) * stock.quantityOwned)
        }
        return stockVal + cash
    }
}
