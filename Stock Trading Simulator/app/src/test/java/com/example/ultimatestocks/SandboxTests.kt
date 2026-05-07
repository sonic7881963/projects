package com.example.ultimatestocks

import com.example.ultimatestocks.entities.NewsEvent
import com.example.ultimatestocks.entities.Sandbox
import com.example.ultimatestocks.entities.StockDetails
import com.example.ultimatestocks.entities.StockInventory
import com.example.ultimatestocks.entities.StockNewPrice
import com.example.ultimatestocks.model.Model
import com.example.ultimatestocks.view.ViewModel
import com.example.ultimatestocks.view.news
import org.junit.Assert.assertEquals
import org.junit.Test
import kotlin.time.Duration.Companion.hours

class SandboxTests {
    @Test
    fun createSandbox() {
        val sandbox = Sandbox(name = "Sample Sandbox", newsEvents = mutableListOf(NewsEvent(
            title = "Test News Event",
            body = "This is a sample news event.",
            newPrices = listOf(StockNewPrice("Ticker", 100.0f, 20.0f))
        )),
            latestNewsNotRevealed = "Test",
            allStocks = listOf(StockDetails("Ticker",
                description = "Description of a stock",
                historicPrice = mutableListOf(100.0f, 200.0f))),
            ownedStocks = mutableListOf(StockInventory(ticker = "Ticker", quantityOwned = 2)),
            portfolioVal = mutableListOf(100.0f, 200.0f),
            cash = 1000.0f
        )
        assertEquals("Sample Sandbox", sandbox.name)
        assertEquals(1, sandbox.newsEvents.size)
        assertEquals("Test", sandbox.latestNewsNotRevealed)
        assertEquals(1, sandbox.allStocks.size)
        assertEquals(1, sandbox.ownedStocks.size)
        assertEquals(1000.0f, sandbox.cash, 0.001f)
        assertEquals(2, sandbox.portfolioVal.size)

        assertEquals(200.0f, sandbox.getStockValue("Ticker"), 0.001f)

        sandbox.buy("Ticker", 1)
        assertEquals(3, sandbox.ownedStocks.get(0).quantityOwned)
        assertEquals(800.0f, sandbox.cash, 0.001f)

        sandbox.sell("Ticker", 2)
        assertEquals(1, sandbox.ownedStocks.get(0).quantityOwned)
        assertEquals(1200.0f, sandbox.cash, 0.001f)


    }

    @Test
    fun createStockNewPrice() {
        val stockNewPrice = StockNewPrice("Ticker", 200.0f, 100.0f)
        assertEquals("Ticker", stockNewPrice.ticker)
        assertEquals(200.0f, stockNewPrice.newPrice, 0.001f)
        assertEquals(100.0f, stockNewPrice.changeInPrice, 0.001f)
    }

    @Test
    fun createNewsEvent() {
        val newsEvent = NewsEvent("News Title", "News Body", listOf(
            StockNewPrice("Ticker", 200.0f, 100.0f),
            StockNewPrice("Ticker 2", 200.0f, 100.0f),

            ))

        assertEquals("News Title", newsEvent.title)
        assertEquals("News Body", newsEvent.body)
        assertEquals(2, newsEvent.newPrices.size)
        assertEquals("Ticker", newsEvent.newPrices.get(0).ticker)
        assertEquals("Ticker 2", newsEvent.newPrices.get(1).ticker)
    }



}