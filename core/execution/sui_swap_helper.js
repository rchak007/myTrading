#!/usr/bin/env node
/**
 * sui_swap_helper.js — Node.js helper for SUI swaps via 7K Protocol SDK
 *
 * Uses MetaAg class from @7kprotocol/sdk-ts which aggregates across
 * Cetus, DeepBook, FlowX, Aftermath, Turbos, and more.
 *
 * Called by Python bot.py as a subprocess:
 *   node sui_swap_helper.js \
 *     --coinIn  "0xdeeb...::deep::DEEP" \
 *     --coinOut "0xdba3...::usdc::USDC" \
 *     --amount  "180000000000" \
 *     --wallet  "0x246c..." \
 *     --privateKey "hex_ed25519_key" \
 *     --slippage 0.03
 *
 * Outputs JSON to stdout:
 *   {"status":"success","digest":"AbCdEf...","amountOut":"5275945629"}
 *   {"status":"error","message":"..."}
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

        // Step 2: Build + sign + execute via swap()
        output_log("Building and executing swap...");

        // Parse the private key
        let keypair;
        if (privateKey.startsWith("suiprivkey")) {
            const { decodeSuiPrivateKey } = await import("@mysten/sui/cryptography");
            const parsed = decodeSuiPrivateKey(privateKey);
            keypair = Ed25519Keypair.fromSecretKey(parsed.secretKey);
        } else {
            const keyBytes = Buffer.from(privateKey.replace("0x", ""), "hex");
            keypair = Ed25519Keypair.fromSecretKey(keyBytes);
        }

        // MetaAg.swap() expects the quote result + signer info
        const swapResult = await ma.swap({
            quote: bestQuote,
            signer: keypair,
            slippage: parseFloat(slippage),
            suiClient: suiClient,
            accountAddress: wallet,
        });

        output_log(`Swap result: ${JSON.stringify(swapResult, null, 0).slice(0, 300)}`);

        // Extract digest from result
        const digest = swapResult?.digest
            || swapResult?.result?.digest
            || swapResult?.txDigest
            || (typeof swapResult === "string" ? swapResult : null);

        if (digest) {
            output({
                status: "success",
                digest: String(digest),
                amountOut: String(expectedOut),
            });
        } else {
            // Maybe swap returned the full tx result
            output({
                status: "success",
                digest: JSON.stringify(swapResult).slice(0, 200),
                amountOut: String(expectedOut),
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