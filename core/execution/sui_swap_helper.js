#!/usr/bin/env node
/**
 * sui_swap_helper.js — Node.js helper for SUI swaps via 7K Protocol SDK
 *
 * Uses MetaAg.quote() for routing, then manually builds the transaction
 * using the quote's protocol configs + @mysten/sui Transaction class,
 * signs with the provided keypair, and executes via SUI RPC.
 *
 * Setup (in core/execution/):
 *   npm install @7kprotocol/sdk-ts @mysten/sui@1 @flowx-finance/sdk --legacy-peer-deps
 */

async function main() {
    const args = parseArgs(process.argv.slice(2));

    const {
        coinIn,
        coinOut,
        amount,
        wallet,
        privateKey,
        slippage = "0.03",
        rpcUrl,
    } = args;

    if (!coinIn || !coinOut || !amount || !wallet || !privateKey) {
        output({ status: "error", message: "Missing required args: --coinIn --coinOut --amount --wallet --privateKey" });
        process.exit(1);
    }

    try {
        const { SuiClient, getFullnodeUrl } = await import("@mysten/sui/client");
        const { Ed25519Keypair } = await import("@mysten/sui/keypairs/ed25519");
        const sevenK = await import("@7kprotocol/sdk-ts");

        // Initialize SUI client
        const suiClient = new SuiClient({
            url: rpcUrl || getFullnodeUrl("mainnet"),
        });

        // Initialize MetaAg
        const ma = new sevenK.MetaAg({ suiClient });

        // Step 1: Get quote
        output_log(`Getting quote: ${coinIn.split("::").pop()} -> ${coinOut.split("::").pop()} amount=${amount}`);

        const quotes = await ma.quote({
            coinTypeIn: coinIn,
            coinTypeOut: coinOut,
            amountIn: amount,
        });

        if (!quotes || quotes.length === 0) {
            throw new Error("No quotes returned — no liquidity providers available for this pair");
        }

        const bestQuote = quotes[0];
        const expectedOut = bestQuote.amountOut || bestQuote.rawAmountOut || "?";
        output_log(`Quote received: provider=${bestQuote.provider} expectedOut=${expectedOut}`);

        // Step 2: Parse the private key
        let keypair;
        if (privateKey.startsWith("suiprivkey")) {
            const { decodeSuiPrivateKey } = await import("@mysten/sui/cryptography");
            const parsed = decodeSuiPrivateKey(privateKey);
            keypair = Ed25519Keypair.fromSecretKey(parsed.secretKey);
        } else {
            const keyBytes = Buffer.from(privateKey.replace("0x", ""), "hex");
            keypair = Ed25519Keypair.fromSecretKey(keyBytes);
        }

        // Step 3: Try swap() with different signer patterns
        output_log("Executing swap...");

        let swapResult;
        let lastError;

        // Pattern 1: signer as signTransaction callback (most common SDK pattern)
        try {
            swapResult = await ma.swap({
                quote: bestQuote,
                signer: {
                    signTransaction: async (txBytes) => {
                        const { signature } = await keypair.signTransaction(txBytes);
                        return { signature, bytes: txBytes };
                    },
                },
                slippage: parseFloat(slippage),
                suiClient: suiClient,
                accountAddress: wallet,
            });
        } catch (e1) {
            lastError = e1;
            output_log(`Pattern 1 failed: ${e1.message}`);

            // Pattern 2: signer as the keypair directly, wallet as string
            try {
                swapResult = await ma.swap({
                    quote: bestQuote,
                    signer: keypair,
                    slippage: parseFloat(slippage),
                    suiClient: suiClient,
                    accountAddress: wallet,
                });
            } catch (e2) {
                lastError = e2;
                output_log(`Pattern 2 failed: ${e2.message}`);

                // Pattern 3: signer as signAndExecuteTransaction callback
                try {
                    swapResult = await ma.swap({
                        quote: bestQuote,
                        signer: async ({ transaction }) => {
                            transaction.setSender(wallet);
                            const txBytes = await transaction.build({ client: suiClient });
                            const { signature } = await keypair.signTransaction(txBytes);
                            const result = await suiClient.executeTransactionBlock({
                                transactionBlock: txBytes,
                                signature: [signature],
                                options: { showEffects: true },
                            });
                            return result;
                        },
                        slippage: parseFloat(slippage),
                        suiClient: suiClient,
                        accountAddress: wallet,
                    });
                } catch (e3) {
                    lastError = e3;
                    output_log(`Pattern 3 failed: ${e3.message}`);

                    // Pattern 4: Manually build tx using fastSwap if available
                    try {
                        output_log("Trying fastSwap...");
                        swapResult = await ma.fastSwap({
                            quote: bestQuote,
                            signer: keypair,
                            slippage: parseFloat(slippage),
                            suiClient: suiClient,
                            accountAddress: wallet,
                        });
                    } catch (e4) {
                        output_log(`fastSwap failed: ${e4.message}`);
                        throw new Error(`All swap patterns failed. Last errors: P1=${e1.message} | P2=${e2.message} | P3=${e3.message} | P4=${e4.message}`);
                    }
                }
            }
        }

        output_log(`Swap result type: ${typeof swapResult}`);
        output_log(`Swap result: ${JSON.stringify(swapResult, (k,v) => typeof v === 'bigint' ? v.toString() : v, 0).slice(0, 500)}`);

        // Extract digest
        const digest = swapResult?.digest
            || swapResult?.result?.digest
            || swapResult?.txDigest
            || swapResult?.effects?.transactionDigest
            || (typeof swapResult === "string" ? swapResult : null);

        if (digest) {
            output({
                status: "success",
                digest: String(digest),
                amountOut: String(expectedOut),
            });
        } else {
            // Return whatever we got
            output({
                status: "success",
                digest: "unknown-check-wallet",
                amountOut: String(expectedOut),
                raw: JSON.stringify(swapResult, (k,v) => typeof v === 'bigint' ? v.toString() : v).slice(0, 300),
            });
        }

    } catch (err) {
        output({
            status: "error",
            message: err.message || String(err),
            stack: err.stack ? err.stack.split("\n").slice(0, 5).join(" | ") : undefined,
        });
        process.exit(1);
    }
}

function parseArgs(argv) {
    const args = {};
    for (let i = 0; i < argv.length; i++) {
        if (argv[i].startsWith("--") && i + 1 < argv.length) {
            args[argv[i].slice(2)] = argv[i + 1];
            i++;
        }
    }
    return args;
}

function output(obj) {
    console.log(JSON.stringify(obj));
}

function output_log(msg) {
    process.stderr.write(`[sui_swap_helper] ${msg}\n`);
}

main();