"use strict";
cc._RF.push(module, '06a79I1MFlKZYzGvjTIzIZ3', 'right1');
// 3.16小游戏/command_TypeScript/level6/right1.ts

"use strict";
// Learn TypeScript:
//  - https://docs.cocos.com/creator/manual/en/scripting/typescript.html
// Learn Attribute:
//  - https://docs.cocos.com/creator/manual/en/scripting/reference/attributes.html
// Learn life-cycle callbacks:
//  - https://docs.cocos.com/creator/manual/en/scripting/life-cycle-callbacks.html
var __extends = (this && this.__extends) || (function () {
    var extendStatics = function (d, b) {
        extendStatics = Object.setPrototypeOf ||
            ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
            function (d, b) { for (var p in b) if (Object.prototype.hasOwnProperty.call(b, p)) d[p] = b[p]; };
        return extendStatics(d, b);
    };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
var _a = cc._decorator, ccclass = _a.ccclass, property = _a.property;
var global___002_1 = require("./global - 002");
var enemy_1 = /** @class */ (function (_super) {
    __extends(enemy_1, _super);
    function enemy_1() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    enemy_1.prototype.onLoad = function () {
        var manager = cc.director.getCollisionManager();
        manager.enabled = true;
    };
    enemy_1.prototype.start = function () {
        this.schedule(function () {
            global___002_1.default.minion_attack = true;
        }, 1);
        this.minion_x = this.node.position.x;
    };
    enemy_1.prototype.onCollisionEnter = function (other, self) {
        cc.log("开始碰撞" + other.tag);
        if (global___002_1.default.minion_attack == true) {
            global___002_1.default.main_hp -= 20;
            var damage = cc.find("Canvas/bj/kuan/enemy_damage");
            damage.getComponent(cc.Label).string = "-20";
            var damage2 = cc.find("Canvas/bj/kuan/xy/main_damage");
            damage2.getComponent(cc.Label).string = "";
        }
    };
    enemy_1.prototype.onCollisionExit = function (other) {
        cc.log("碰撞结束");
        global___002_1.default.minion_attack = false;
        if (global___002_1.default.main_hp <= 0) {
            other.node.active = false;
            var lose = cc.find("Canvas/bj/fail");
            lose.active = true;
        }
        else {
            other.node.children[0].setContentSize(global___002_1.default.main_hp, 19);
        }
    };
    enemy_1.prototype.update = function (dt) {
        if (global___002_1.default.minion_attack == true) {
            this.node.setPosition(this.node.position.x - 1000 * dt, this.node.position.y);
        }
        else {
            if (!(this.node.position.x >= this.minion_x + 50) && (this.node.position.x <= this.minion_x - 50)) {
                this.node.setPosition(this.node.position.x + 1000 * dt, this.node.position.y);
            }
        }
    };
    enemy_1 = __decorate([
        ccclass
    ], enemy_1);
    return enemy_1;
}(cc.Component));
exports.default = enemy_1;

cc._RF.pop();